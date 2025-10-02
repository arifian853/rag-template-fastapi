# Tambahkan import ini di bagian atas
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Response
import pandas as pd
import PyPDF2
from pathlib import Path
import io
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import csv
import google.generativeai as genai
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
import motor.motor_asyncio
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import uvicorn
import io
import asyncio
import cloudinary
import cloudinary.uploader
import cloudinary.api
import cloudinary.utils
from datetime import datetime
import base64
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI, server_api=ServerApi('1'))
db = client.get_database("chatbot_cs")
knowledge_collection = db.get_collection("rag_data_knowledge")
system_prompt_collection = db.get_collection("system_prompts")
files_collection = db.get_collection("files")  # New collection for file metadata

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = []

class KnowledgeItem(BaseModel):
    title: str
    content: str
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class UpdateKnowledgeItem(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SystemPrompt(BaseModel):
    name: str
    prompt: str
    is_active: bool = False
    is_default: bool = False
    description: Optional[str] = None

class UpdateSystemPrompt(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None
    is_active: Optional[bool] = None
    is_default: Optional[bool] = None
    description: Optional[str] = None

class FileMetadata(BaseModel):
    filename: str
    original_filename: str
    cloudinary_url: str
    cloudinary_public_id: str
    file_type: str  # pdf, excel, csv
    file_size: int
    upload_date: Optional[datetime] = None
    knowledge_ids: List[str] = []
    metadata: Optional[Dict[str, Any]] = None

class ChatWithFileRequest(BaseModel):
    message: str
    file_id: str
    conversation_history: List[Dict[str, str]] = []

# Helper functions
async def generate_embedding(text: str):
    """Generate embedding for text using SentenceTransformer"""
    embedding = model.encode(text)
    return embedding.tolist()

async def add_knowledge(knowledge: KnowledgeItem):
    """Add knowledge to database with embedding"""
    embedding = await generate_embedding(knowledge.content)
    document = {
        "title": knowledge.title,
        "content": knowledge.content,
        "source": knowledge.source,
        "metadata": knowledge.metadata,
        "embedding": embedding
    }
    result = await knowledge_collection.insert_one(document)
    return result.inserted_id

async def search_similar_documents(query: str, limit: int = 5):
    """Search for similar documents by computing similarity in the application"""
    query_embedding = await generate_embedding(query)
    
    # Fetch all documents (consider pagination for large collections)
    documents = []
    async for doc in knowledge_collection.find({}): 
        documents.append(doc)
    
    # Calculate cosine similarity for each document
    results_with_scores = []
    for doc in documents:
        # Skip documents without embeddings
        if "embedding" not in doc or not doc["embedding"]:
            continue
            
        # Calculate cosine similarity
        doc_embedding = doc["embedding"]
        similarity = cosine_similarity(query_embedding, doc_embedding)
        
        # Add document with similarity score
        doc_with_score = {
            "_id": doc["_id"],
            "title": doc["title"],
            "content": doc["content"],
            "source": doc.get("source"),
            "metadata": doc.get("metadata", {}),
            "score": similarity
        }
        results_with_scores.append(doc_with_score)
    
    # Sort by similarity score (highest first) and limit results
    results_with_scores.sort(key=lambda x: x["score"], reverse=True)
    top_results = results_with_scores[:limit]
    
    return top_results

# Tambahkan fungsi helper untuk menghitung cosine similarity
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    # Convert to numpy arrays if they aren't already
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Calculate dot product and magnitudes
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    # Prevent division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    return float(similarity)

# Helper functions untuk system prompt
async def get_active_system_prompt():
    """Get the currently active system prompt"""
    prompt_doc = await system_prompt_collection.find_one({"is_active": True})
    if prompt_doc:
        return prompt_doc["prompt"]
    
    # Default system prompt jika tidak ada yang aktif
    return "Based on the following information, please answer the user's question. If you don't know the answer based on the provided context, say so. Don't act like a robot, don't act like a chatbot, act like a simple human being, not an introverted nor extroverted person, just decent being. Use Indonesian language as your main language but if the user question is in English, please answer with English, don't answer with another language beside Indonesian and English."

async def process_chat(message: str, history: List[Dict[str, str]]):
    """Process chat message with RAG approach"""
    # Search for relevant context
    relevant_docs = await search_similar_documents(message)
    
    # Format context for the model
    context = "\n\n".join([f"Title: {doc['title']}\nContent: {doc['content']}" for doc in relevant_docs])
    
    # Format conversation history
    formatted_history = []
    for entry in history:
        if 'user' in entry:
            formatted_history.append({"role": "user", "parts": [entry['user']]})
        if 'assistant' in entry:
            formatted_history.append({"role": "model", "parts": [entry['assistant']]})
    
    # Create Gemini model
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Start conversation
    chat = model.start_chat(history=formatted_history)
    
    # Get active system prompt from database
    system_prompt = await get_active_system_prompt()
    
    # Generate response with context
    prompt = f"""
    {system_prompt}

    Context:
    {context}

    User question: {message}
"""

    response = await asyncio.to_thread(lambda: chat.send_message(prompt))
    
    return {
        "response": response.text,
        "sources": [{
            "_id": str(doc["_id"]),
            "title": doc["title"], 
            "content": doc["content"], 
            "source": doc["source"],
            "similarity_score": round(doc["score"], 4)
        } for doc in relevant_docs]
    }

# Routes
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for frontend integration"""
    try:
        result = await process_chat(request.message, request.history)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add pagination models
class PaginationParams(BaseModel):
    page: int = 1
    limit: int = 15
    sort_order: str = "newest"  # newest or oldest

class PaginatedResponse(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_prev: bool

@app.get("/knowledge", response_model=PaginatedResponse)
async def list_knowledge(
    page: int = 1,
    limit: int = 15,
    sort_order: str = "newest"
):
    """List knowledge items with pagination"""
    # Validate parameters
    if page < 1:
        page = 1
    if limit < 1 or limit > 100:  # Max 100 items per page
        limit = 15
    
    # Calculate skip value
    skip = (page - 1) * limit
    
    # Determine sort direction
    sort_direction = -1 if sort_order == "newest" else 1
    
    # Get total count
    total = await knowledge_collection.count_documents({})
    
    # Get paginated items
    items = []
    cursor = knowledge_collection.find(
        {}, 
        {"embedding": 0}
    ).sort("_id", sort_direction).skip(skip).limit(limit)
    
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        items.append(doc)
    
    # Calculate pagination info
    total_pages = (total + limit - 1) // limit  # Ceiling division
    has_next = page < total_pages
    has_prev = page > 1
    
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages,
        has_next=has_next,
        has_prev=has_prev
    )

# Keep the old endpoint for backward compatibility (optional)
@app.get("/knowledge/all", response_model=List[Dict[str, Any]])
async def list_all_knowledge():
    """List all knowledge items (for backward compatibility)"""
    items = []
    async for doc in knowledge_collection.find({}, {"embedding": 0}):
        doc["_id"] = str(doc["_id"])
        items.append(doc)
    return items

@app.post("/add-knowledge")
async def create_knowledge(knowledge: KnowledgeItem):
    """Add a new knowledge item"""
    result = await add_knowledge(knowledge)
    return {"id": str(result), "message": "Knowledge added successfully"}

@app.get("/knowledge/{knowledge_id}")
async def get_knowledge_by_id(knowledge_id: str):
    """Get a specific knowledge item by ID"""
    try:
        doc = await knowledge_collection.find_one({"_id": ObjectId(knowledge_id)}, {"embedding": 0})
        if not doc:
            raise HTTPException(status_code=404, detail="Knowledge not found")
        
        doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {str(e)}")

@app.put("/knowledge/{knowledge_id}")
async def update_knowledge(knowledge_id: str, knowledge: UpdateKnowledgeItem):
    """Update a knowledge item"""
    try:
        # Prepare update data
        update_data = {}
        if knowledge.title is not None:
            update_data["title"] = knowledge.title
        if knowledge.content is not None:
            update_data["content"] = knowledge.content
            # Regenerate embedding if content is updated
            update_data["embedding"] = await generate_embedding(knowledge.content)
        if knowledge.source is not None:
            update_data["source"] = knowledge.source
        if knowledge.metadata is not None:
            update_data["metadata"] = knowledge.metadata
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        result = await knowledge_collection.update_one(
            {"_id": ObjectId(knowledge_id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Knowledge not found")
        
        return {"message": "Knowledge updated successfully"}
    except Exception as e:
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid ID format")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/knowledge/{knowledge_id}")
async def delete_knowledge(knowledge_id: str):
    """Delete a knowledge item"""
    try:
        result = await knowledge_collection.delete_one({"_id": ObjectId(knowledge_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Knowledge not found")
        
        return {"message": "Knowledge deleted successfully"}
    except Exception as e:
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid ID format")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/knowledge")
async def delete_all_knowledge():
    """Delete all knowledge items"""
    try:
        result = await knowledge_collection.delete_many({})
        return {
            "message": f"All knowledge deleted successfully. {result.deleted_count} items removed.",
            "deleted_count": result.deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions untuk file processing
async def extract_text_from_pdf(file_content: bytes) -> tuple[str, str]:
    """Extract text from PDF and get title if available"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Try to get title from metadata
        title = None
        if pdf_reader.metadata:
            title = pdf_reader.metadata.get('/Title')
        
        return text.strip(), title
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

async def process_excel_file(file_content: bytes, filename: str) -> List[Dict[str, Any]]:
    """Process Excel file and return list of knowledge items"""
    try:
        # Read Excel file
        excel_file = io.BytesIO(file_content)
        df = pd.read_excel(excel_file)
        
        knowledge_items = []
        
        # Convert each row to knowledge item
        for index, row in df.iterrows():
            # Create content from all columns
            content_parts = []
            for col, value in row.items():
                if pd.notna(value):
                    content_parts.append(f"{col}: {value}")
            
            content = "\n".join(content_parts)
            title = f"{filename} - Row {index + 1}"
            
            # Try to use first column as title if it looks like a title
            first_col_value = row.iloc[0] if len(row) > 0 and pd.notna(row.iloc[0]) else None
            if first_col_value and len(str(first_col_value)) < 100:
                title = str(first_col_value)
            
            knowledge_items.append({
                "title": title,
                "content": content,
                "source": filename,
                "metadata": {
                    "file_type": "excel",
                    "filename": filename,
                    "row_index": index + 1,
                    "columns": list(df.columns)
                }
            })
        
        return knowledge_items
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing Excel file: {str(e)}")

async def process_csv_file(file_content: bytes, filename: str) -> List[Dict[str, Any]]:
    """Process CSV file and return list of knowledge items"""
    try:
        # Read CSV file
        csv_text = file_content.decode('utf-8')
        csv_file = io.StringIO(csv_text)
        csv_reader = csv.DictReader(csv_file)
        
        knowledge_items = []
        
        for index, row in enumerate(csv_reader):
            # Create content from all columns
            content_parts = []
            for key, value in row.items():
                if value and value.strip():
                    content_parts.append(f"{key}: {value}")
            
            content = "\n".join(content_parts)
            title = f"{filename} - Row {index + 1}"
            
            # Try to use first column as title if it looks like a title
            first_value = list(row.values())[0] if row.values() else None
            if first_value and len(first_value) < 100:
                title = first_value
            
            knowledge_items.append({
                "title": title,
                "content": content,
                "source": filename,
                "metadata": {
                    "file_type": "csv",
                    "filename": filename,
                    "row_index": index + 1,
                    "columns": list(row.keys())
                }
            })
        
        return knowledge_items
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")

# fungsi OCR
async def extract_text_with_ocr(pdf_content: bytes, filename: str):
    """Extract text from PDF using OCR for scanned documents"""
    try:
        print(f"Starting OCR processing for {filename}")
        
        # Convert PDF pages to images
        images = convert_from_bytes(pdf_content, dpi=300, fmt='jpeg')
        print(f"Converted PDF to {len(images)} images")
        
        extracted_texts = []
        
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}")
            
            # Use Tesseract OCR to extract text
            # Configure for Indonesian and English
            custom_config = r'--oem 3 --psm 6 -l ind+eng'
            
            try:
                text = pytesseract.image_to_string(image, config=custom_config)
                if text.strip():  # Only add if text found
                    extracted_texts.append(f"Page {i+1}:\n{text.strip()}")
                    print(f"Page {i+1}: Found {len(text.strip())} characters")
                else:
                    print(f"Page {i+1}: No text detected")
            except Exception as ocr_error:
                print(f"OCR error on page {i+1}: {str(ocr_error)}")
                # Fallback: try with different PSM mode
                try:
                    fallback_config = r'--oem 3 --psm 3 -l eng'
                    text = pytesseract.image_to_string(image, config=fallback_config)
                    if text.strip():
                        extracted_texts.append(f"Page {i+1} (fallback):\n{text.strip()}")
                except:
                    extracted_texts.append(f"Page {i+1}: [OCR failed - image content]")
        
        full_text = "\n\n".join(extracted_texts) if extracted_texts else ""
        print(f"OCR completed. Total text length: {len(full_text)}")
        
        return full_text
        
    except Exception as e:
        print(f"OCR processing failed: {str(e)}")
        return f"[OCR Error: {str(e)} - This appears to be a scanned document that couldn't be processed]"

# File Upload Endpoints
@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload file, save to Cloudinary, and create knowledge entries with OCR support"""
    try:
        file_content = await file.read()
        file_extension = file.filename.split('.')[-1].lower()
        
        # Upload to Cloudinary first
        cloudinary_data = await upload_file_to_cloudinary(file_content, file.filename, file_extension)
        
        # Process file content for knowledge creation
        knowledge_ids = []
        
        if file_extension == 'pdf':
            # Try normal PDF text extraction first
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"
            
            # Check if we got meaningful text
            meaningful_text = full_text.strip().replace('\n', ' ').replace(' ', '')
            
            if len(meaningful_text) < 50:  # Threshold for "no meaningful text"
                print(f"PDF appears to be scanned/image-based. Attempting OCR...")
                
                # Fallback to OCR
                ocr_text = await extract_text_with_ocr(file_content, file.filename)
                
                if ocr_text and len(ocr_text.strip()) > 50:
                    full_text = f"[OCR Extracted Content]\n\n{ocr_text}"
                    print(f"OCR successful: {len(ocr_text)} characters extracted")
                else:
                    full_text = f"[Scanned Document - Limited Text Extraction]\n\nThis appears to be a scanned document or image-based PDF. Text extraction was attempted but may be incomplete.\n\nOriginal filename: {file.filename}\nPages: {len(pdf_reader.pages)}\n\nNote: This document may contain visual content (images, charts, diagrams) that cannot be processed as text."
                    print("OCR failed or returned minimal text")
            else:
                print(f"PDF text extraction successful: {len(meaningful_text)} characters")
            
            # Create knowledge entry
            knowledge = KnowledgeItem(
                title=f"PDF: {file.filename}",
                content=full_text,
                source=file.filename,
                metadata={
                    "file_type": "pdf",
                    "filename": file.filename,
                    "pages": len(pdf_reader.pages),
                    "cloudinary_url": cloudinary_data["url"],
                    "processing_method": "ocr" if "OCR Extracted" in full_text else "standard",
                    "text_length": len(full_text)
                }
            )
            result = await add_knowledge(knowledge)
            knowledge_ids.append(str(result))
            
        elif file_extension in ['xlsx', 'xls']:
            # Excel processing remains the same
            excel_file = io.BytesIO(file_content)
            df = pd.read_excel(excel_file)
            
            for index, row in df.iterrows():
                row_content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                knowledge = KnowledgeItem(
                    title=f"Excel Row {index + 1}: {file.filename}",
                    content=row_content,
                    source=file.filename,
                    metadata={
                        "file_type": "excel",
                        "filename": file.filename,
                        "row_index": index + 1,
                        "cloudinary_url": cloudinary_data["url"]
                    }
                )
                result = await add_knowledge(knowledge)
                knowledge_ids.append(str(result))
                
        elif file_extension == 'csv':
            # CSV processing remains the same
            csv_text = file_content.decode('utf-8')
            csv_file = io.StringIO(csv_text)
            csv_reader = csv.DictReader(csv_file)
            
            for index, row in enumerate(csv_reader):
                row_content = " | ".join([f"{key}: {value}" for key, value in row.items() if value])
                knowledge = KnowledgeItem(
                    title=f"CSV Row {index + 1}: {file.filename}",
                    content=row_content,
                    source=file.filename,
                    metadata={
                        "file_type": "csv",
                        "filename": file.filename,
                        "row_index": index + 1,
                        "cloudinary_url": cloudinary_data["url"]
                    }
                )
                result = await add_knowledge(knowledge)
                knowledge_ids.append(str(result))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Save file metadata to database
        file_id = await save_file_metadata(
            filename=file.filename,
            original_filename=file.filename,
            cloudinary_data=cloudinary_data,
            file_type=file_extension,
            knowledge_ids=knowledge_ids
        )
        
        processing_info = ""
        if file_extension == 'pdf':
            if "OCR Extracted" in full_text:
                processing_info = " (OCR processed for scanned content)"
            elif "Limited Text Extraction" in full_text:
                processing_info = " (Scanned document - limited text extraction)"
        
        return {
            "message": f"File uploaded successfully to Cloudinary and processed for knowledge{processing_info}",
            "file_id": file_id,
            "cloudinary_url": cloudinary_data["url"],
            "knowledge_ids": knowledge_ids,
            "items_created": len(knowledge_ids),
            "processing_note": processing_info.strip()
        }
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/files/{file_id}/reprocess-ocr")
async def reprocess_with_ocr(file_id: str):
    """Re-process PDF file with OCR if initial processing failed"""
    try:
        # Get file metadata
        file_doc = await files_collection.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        if file_doc.get("file_type") != "pdf":
            raise HTTPException(status_code=400, detail="OCR reprocessing only available for PDF files")
        
        # Get PDF content from base64 storage
        import requests
        response = requests.get(file_doc["cloudinary_url"])
        
        if response.status_code == 200:
            content = response.text
            if content.startswith("PDF_DATA:"):
                pdf_base64 = content[9:]
                pdf_bytes = base64.b64decode(pdf_base64)
                
                # Process with OCR
                ocr_text = await extract_text_with_ocr(pdf_bytes, file_doc["original_filename"])
                
                if ocr_text and len(ocr_text.strip()) > 50:
                    # Update existing knowledge entry
                    if file_doc.get("knowledge_ids"):
                        knowledge_id = file_doc["knowledge_ids"][0]
                        
                        updated_content = f"[OCR Re-processed Content]\n\n{ocr_text}"
                        
                        # Generate new embedding
                        embedding = await generate_embedding(updated_content)
                        
                        # Update knowledge
                        await knowledge_collection.update_one(
                            {"_id": ObjectId(knowledge_id)},
                            {"$set": {
                                "content": updated_content,
                                "embedding": embedding,
                                "metadata.processing_method": "ocr_reprocessed",
                                "metadata.reprocessed_at": datetime.now().isoformat()
                            }}
                        )
                        
                        return {
                            "message": "PDF reprocessed with OCR successfully",
                            "text_length": len(ocr_text),
                            "knowledge_updated": True
                        }
                    else:
                        raise HTTPException(status_code=404, detail="No knowledge entries found for this file")
                else:
                    raise HTTPException(status_code=422, detail="OCR processing failed or returned minimal text")
            else:
                raise HTTPException(status_code=400, detail="Invalid PDF storage format")
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch PDF content")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint untuk upload dengan opsi kustomisasi
@app.post("/upload-csv-custom")
async def upload_csv_custom(
    file: UploadFile = File(...),
    title_column: str = Form(...),
    content_column: str = Form(...)
):
    """Upload CSV with custom column mapping"""
    try:
        file_content = await file.read()
        csv_text = file_content.decode('utf-8')
        csv_file = io.StringIO(csv_text)
        csv_reader = csv.DictReader(csv_file)
        
        results = []
        for row in csv_reader:
            if title_column in row and content_column in row:
                knowledge = KnowledgeItem(
                    title=row[title_column],
                    content=row[content_column],
                    source=file.filename,
                    metadata={
                        "file_type": "csv_custom",
                        "filename": file.filename,
                        "title_column": title_column,
                        "content_column": content_column,
                        "row_data": row
                    }
                )
                result = await add_knowledge(knowledge)
                results.append(str(result))
        
        return {
            "message": f"CSV file processed with custom mapping",
            "ids": results,
            "items_created": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-excel-custom")
async def upload_excel_custom(
    file: UploadFile = File(...),
    title_column: str = Form(...),
    content_column: str = Form(...)
):
    """Upload Excel with custom column mapping"""
    try:
        file_content = await file.read()
        excel_file = io.BytesIO(file_content)
        df = pd.read_excel(excel_file)
        
        if title_column not in df.columns or content_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Columns '{title_column}' or '{content_column}' not found in Excel file"
            )
        
        results = []
        for index, row in df.iterrows():
            if pd.notna(row[title_column]) and pd.notna(row[content_column]):
                knowledge = KnowledgeItem(
                    title=str(row[title_column]),
                    content=str(row[content_column]),
                    source=file.filename,
                    metadata={
                        "file_type": "excel_custom",
                        "filename": file.filename,
                        "title_column": title_column,
                        "content_column": content_column,
                        "row_index": index + 1,
                        "row_data": row.to_dict()
                    }
                )
                result = await add_knowledge(knowledge)
                results.append(str(result))
        
        return {
            "message": f"Excel file processed with custom mapping",
            "ids": results,
            "items_created": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)


# System Prompt Endpoints
@app.get("/system-prompts")
async def list_system_prompts():
    """List all system prompts"""
    prompts = []
    async for doc in system_prompt_collection.find({}):
        doc["_id"] = str(doc["_id"])
        prompts.append(doc)
    return prompts

@app.get("/system-prompts/active")
async def get_active_prompt():
    """Get the currently active system prompt"""
    prompt_doc = await system_prompt_collection.find_one({"is_active": True})
    if not prompt_doc:
        raise HTTPException(status_code=404, detail="No active system prompt found")
    
    prompt_doc["_id"] = str(prompt_doc["_id"])
    return prompt_doc

@app.post("/system-prompts")
async def create_system_prompt(prompt: SystemPrompt):
    """Create a new system prompt"""
    try:
        # If this prompt is set as active, deactivate all others
        if prompt.is_active:
            await system_prompt_collection.update_many(
                {"is_active": True},
                {"$set": {"is_active": False}}
            )
        
        document = {
            "name": prompt.name,
            "prompt": prompt.prompt,
            "is_active": prompt.is_active,
            "description": prompt.description,
            "created_at": asyncio.get_event_loop().time()
        }
        
        result = await system_prompt_collection.insert_one(document)
        return {"id": str(result.inserted_id), "message": "System prompt created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/system-prompts/{prompt_id}")
async def update_system_prompt(prompt_id: str, prompt: UpdateSystemPrompt):
    """Update a system prompt"""
    try:
        update_data = {}
        
        if prompt.name is not None:
            update_data["name"] = prompt.name
        if prompt.prompt is not None:
            update_data["prompt"] = prompt.prompt
        if prompt.description is not None:
            update_data["description"] = prompt.description
        if prompt.is_active is not None:
            update_data["is_active"] = prompt.is_active
            
            # If setting this as active, deactivate all others
            if prompt.is_active:
                await system_prompt_collection.update_many(
                    {"_id": {"$ne": ObjectId(prompt_id)}, "is_active": True},
                    {"$set": {"is_active": False}}
                )
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        result = await system_prompt_collection.update_one(
            {"_id": ObjectId(prompt_id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="System prompt not found")
        
        return {"message": "System prompt updated successfully"}
    except Exception as e:
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid ID format")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/system-prompts/{prompt_id}")
async def delete_system_prompt(prompt_id: str):
    """Delete a system prompt"""
    try:
        # First, check if the prompt to be deleted exists and if it's active
        prompt_to_delete = await system_prompt_collection.find_one({"_id": ObjectId(prompt_id)})
        
        if not prompt_to_delete:
            raise HTTPException(status_code=404, detail="System prompt not found")
        
        # Check if it's a default prompt (cannot be deleted)
        if prompt_to_delete.get("is_default", False):
            raise HTTPException(status_code=400, detail="Cannot delete default prompt")
        
        # Check if the prompt to be deleted is currently active or not
        is_active_prompt = prompt_to_delete.get("is_active", False)
        
        # Delete the prompt
        result = await system_prompt_collection.delete_one({"_id": ObjectId(prompt_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="System prompt not found")
        
        # If we deleted an active prompt, activate the default prompt
        if is_active_prompt:
            # Find the default prompt
            default_prompt = await system_prompt_collection.find_one({"is_default": True})
            
            if default_prompt:
                # Activate the default prompt
                await system_prompt_collection.update_one(
                    {"_id": default_prompt["_id"]},
                    {"$set": {"is_active": True}}
                )
            else:
                # If no default prompt exists, create and activate it
                await initialize_default_prompt()
        
        return {"message": "System prompt deleted successfully"}
    except Exception as e:
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid ID format")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system-prompts/{prompt_id}/activate")
async def activate_system_prompt(prompt_id: str):
    """Activate a specific system prompt"""
    try:
        # Deactivate all prompts first
        await system_prompt_collection.update_many(
            {"is_active": True},
            {"$set": {"is_active": False}}
        )
        
        # Activate the specified prompt
        result = await system_prompt_collection.update_one(
            {"_id": ObjectId(prompt_id)},
            {"$set": {"is_active": True}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="System prompt not found")
        
        return {"message": "System prompt activated successfully"}
    except Exception as e:
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid ID format")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system-prompts/initialize")
async def initialize_default_prompt():
    """Initialize default system prompt if none exists"""
    try:
        # Check if any prompt exists
        existing_count = await system_prompt_collection.count_documents({})
        
        if existing_count == 0:
            default_prompt = {
                "name": "Default RAG chatbot",
                "prompt": "Based on the following information, please answer the user's question. If you don't know the answer based on the provided context, say so. Don't act like a robot, don't act like a chatbot, act like a simple human being, not an introverted nor extroverted person, just decent being. Use Indonesian language as your main language but if the user question is in English, please answer with English, don't answer with another language beside Indonesian and English.",
                "is_active": True,
                "is_default": True,
                "description": "Default RAG chatbot system prompt",
                "created_at": asyncio.get_event_loop().time()
            }
            
            result = await system_prompt_collection.insert_one(default_prompt)
            return {"message": "Default system prompt initialized", "id": str(result.inserted_id)}
        else:
            return {"message": "System prompts already exist"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system-prompts/reset-to-default")
async def reset_to_default():
    """Reset system to use default prompt"""
    try:
        # Find the default prompt
        default_prompt = await system_prompt_collection.find_one({"is_default": True})
        
        if not default_prompt:
            # If no default prompt exists, initialize it first
            await initialize_default_prompt()
            default_prompt = await system_prompt_collection.find_one({"is_default": True})
        
        if not default_prompt:
            raise HTTPException(status_code=404, detail="Default prompt not found")
        
        # Deactivate all prompts first
        await system_prompt_collection.update_many(
            {"is_active": True},
            {"$set": {"is_active": False}}
        )
        
        # Activate the default prompt
        await system_prompt_collection.update_one(
            {"_id": default_prompt["_id"]},
            {"$set": {"is_active": True}}
        )
        
        return {"message": "Reset to default prompt successfully"}
    except Exception as e:
        print(f"Error in reset_to_default: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))

# Helper function for file upload to Cloudinary
async def upload_file_to_cloudinary(file_content: bytes, filename: str, file_type: str):
    """Upload file to Cloudinary with PDF workaround"""
    try:
        if file_type.lower() == 'pdf':
            # Upload PDF as base64 text file menggunakan BytesIO
            pdf_base64 = base64.b64encode(file_content).decode('utf-8')
            text_content = f"PDF_BASE64_START\n{pdf_base64}\nPDF_BASE64_END"
            
            # Convert string to bytes untuk upload
            text_bytes = text_content.encode('utf-8')
            text_io = io.BytesIO(text_bytes)
            
            result = cloudinary.uploader.upload(
                text_io,
                resource_type="raw",
                public_id=f"chatbot_files/pdf_b64_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename.replace('.pdf', '.txt')}",
                use_filename=False,
                unique_filename=True,
                format="txt"  # Explicitly set as text file
            )
            
            return {
                "url": result["secure_url"],
                "public_id": result["public_id"],
                "bytes": len(file_content),
                "storage_type": "pdf_as_base64"
            }
        else:
            # Normal upload untuk file lain
            resource_type = "raw" if file_type.lower() in ['xlsx', 'xls', 'csv'] else "auto"
            
            result = cloudinary.uploader.upload(
                file_content,
                resource_type=resource_type,
                public_id=f"chatbot_files/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}",
                use_filename=True,
                unique_filename=True
            )
            
            return {
                "url": result["secure_url"],
                "public_id": result["public_id"],
                "bytes": result.get("bytes", len(file_content)),
                "storage_type": "normal"
            }
            
    except Exception as e:
        print(f"Cloudinary upload error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file to Cloudinary: {str(e)}")

# Helper function to save file metadata
async def save_file_metadata(filename: str, original_filename: str, cloudinary_data: dict, file_type: str, knowledge_ids: List[str] = None):
    """Save file metadata to database"""
    try:
        file_doc = {
            "filename": filename,
            "original_filename": original_filename,
            "cloudinary_url": cloudinary_data["url"],
            "cloudinary_public_id": cloudinary_data["public_id"],
            "file_type": file_type,
            "file_size": cloudinary_data["bytes"],
            "upload_date": datetime.now(),
            "knowledge_ids": knowledge_ids or [],
            "metadata": {
                "storage_type": cloudinary_data.get("storage_type", "normal")
            }
        }
        
        result = await files_collection.insert_one(file_doc)
        return str(result.inserted_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file metadata: {str(e)}")

# Tambah endpoint untuk serve PDF dari base64
@app.get("/files/{file_id}/pdf")
async def serve_pdf(file_id: str):
    """Serve PDF from base64 storage"""
    try:
        file_doc = await files_collection.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if it's a PDF stored as base64
        if (file_doc.get("file_type") == "pdf" and 
            file_doc.get("metadata", {}).get("storage_type") == "pdf_as_base64"):
            
            # Fetch base64 content from Cloudinary
            import requests
            try:
                response = requests.get(file_doc["cloudinary_url"])
                
                if response.status_code == 200:
                    content = response.text
                    
                    # Extract base64 content - simplified parsing
                    if content.startswith("PDF_DATA:"):
                        pdf_base64 = content[9:]  # Remove "PDF_DATA:" prefix
                    elif "PDF_BASE64_START" in content:
                        # Fallback untuk format lama
                        start_marker = "PDF_BASE64_START\n"
                        end_marker = "\nPDF_BASE64_END"
                        start_idx = content.find(start_marker) + len(start_marker)
                        end_idx = content.find(end_marker)
                        pdf_base64 = content[start_idx:end_idx].strip()
                    else:
                        # Assume entire content is base64
                        pdf_base64 = content.strip()
                    
                    try:
                        pdf_bytes = base64.b64decode(pdf_base64)
                        
                        return Response(
                            content=pdf_bytes,
                            media_type="application/pdf",
                            headers={
                                "Content-Disposition": f"inline; filename={file_doc['original_filename']}",
                                "Content-Type": "application/pdf"
                            }
                        )
                    except Exception as decode_error:
                        print(f"Base64 decode error: {str(decode_error)}")
                        raise HTTPException(status_code=500, detail="Failed to decode PDF content")
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to fetch from Cloudinary: {response.status_code}")
                    
            except requests.RequestException as req_error:
                print(f"Request error: {str(req_error)}")
                raise HTTPException(status_code=500, detail="Failed to fetch PDF content")
        else:
            # For normal files, redirect to Cloudinary URL
            return {"redirect_url": file_doc["cloudinary_url"]}
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Serve PDF error: {str(e)}")
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid file ID format")
        raise HTTPException(status_code=500, detail=str(e))

# Chat with File Endpoint
@app.post("/chat-with-file")
async def chat_with_file(request: ChatWithFileRequest):
    """Chat with specific file context"""
    try:
        print(f"DEBUG: Received chat request for file_id: {request.file_id}")
        
        # Get file metadata
        file_doc = await files_collection.find_one({"_id": ObjectId(request.file_id)})
        if not file_doc:
            print(f"DEBUG: File not found with ID: {request.file_id}")
            raise HTTPException(status_code=404, detail="File not found")
        
        print(f"DEBUG: Found file: {file_doc.get('original_filename')}")
        
        # Get knowledge entries for this file
        knowledge_docs = []
        if file_doc.get("knowledge_ids"):
            print(f"DEBUG: Found {len(file_doc['knowledge_ids'])} knowledge IDs")
            for knowledge_id in file_doc["knowledge_ids"]:
                try:
                    doc = await knowledge_collection.find_one({"_id": ObjectId(knowledge_id)})
                    if doc:
                        knowledge_docs.append(doc)
                        print(f"DEBUG: Added knowledge doc: {doc.get('title', 'No title')}")
                except Exception as e:
                    print(f"Warning: Failed to get knowledge {knowledge_id}: {str(e)}")
        
        if not knowledge_docs:
            print("DEBUG: No knowledge documents found")
            raise HTTPException(status_code=404, detail="No knowledge found for this file")
        
        print(f"DEBUG: Processing {len(knowledge_docs)} knowledge documents")
        
        # Search within file's knowledge only
        query_embedding = await generate_embedding(request.message)
        print(f"DEBUG: Generated query embedding with length: {len(query_embedding)}")
        
        # Calculate similarity only with this file's knowledge
        results_with_scores = []
        for doc in knowledge_docs:
            if "embedding" not in doc or not doc["embedding"]:
                print(f"DEBUG: Skipping doc without embedding: {doc.get('title', 'No title')}")
                continue
            
            try:
                similarity = cosine_similarity(query_embedding, doc["embedding"])
                results_with_scores.append((doc, similarity))
                print(f"DEBUG: Calculated similarity {similarity:.3f} for doc: {doc.get('title', 'No title')}")
            except Exception as e:
                print(f"DEBUG: Error calculating similarity: {str(e)}")
                continue
        
        # Sort by similarity and get top results
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = results_with_scores[:5]  # Top 5 most relevant
        
        print(f"DEBUG: Found {len(top_results)} relevant results")
        
        # Format context from file
        context_parts = []
        for doc, score in top_results:
            if score > 0.3:  # Similarity threshold
                context_parts.append(f"Content: {doc['content']}")
                print(f"DEBUG: Added context with score {score:.3f}")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant content found in the file."
        print(f"DEBUG: Final context length: {len(context)} characters")
        
        # Get active system prompt
        try:
            system_prompt = await get_active_system_prompt()
            print("DEBUG: Retrieved system prompt successfully")
        except Exception as e:
            print(f"DEBUG: Error getting system prompt: {str(e)}")
            system_prompt = "You are a helpful assistant."
        
        # Format conversation history
        conversation_context = ""
        if request.conversation_history:
            for msg in request.conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                conversation_context += f"{role}: {content}\n"
        
        print(f"DEBUG: Conversation context length: {len(conversation_context)} characters")
        
        # Create file-specific prompt
        file_prompt = f"""
{system_prompt}

IMPORTANT: You are now chatting specifically about the file "{file_doc['original_filename']}" ({file_doc['file_type'].upper()}).
Focus your responses on the content of this file. Use the provided context from the file to answer questions.

File Information:
- Filename: {file_doc['original_filename']}
- Type: {file_doc['file_type'].upper()}
- Upload Date: {file_doc.get('upload_date', 'Unknown')}

Context from file:
{context}

Previous conversation:
{conversation_context}

User question: {request.message}

Please answer based on the file content provided above. If the question cannot be answered from the file content, mention that the information is not available in this specific file.
"""

        print(f"DEBUG: Generated prompt with length: {len(file_prompt)} characters")
        
        # Generate response using Gemini
        try:
            gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            response = gemini_model.generate_content(file_prompt)
            print("DEBUG: Generated response from Gemini successfully")
            
            return {
                "response": response.text,
                "file_name": file_doc['original_filename'],
                "file_type": file_doc['file_type'],
                "context_used": len(context_parts),
                "sources_found": len(top_results)
            }
        except Exception as e:
            print(f"DEBUG: Error generating Gemini response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"DEBUG: Unexpected error in chat_with_file: {str(e)}")
        print(f"DEBUG: Error type: {type(e).__name__}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid file ID format")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# File Management Endpoints
@app.get("/files")
async def list_files():
    """List all uploaded files"""
    try:
        files = []
        async for doc in files_collection.find({}).sort("upload_date", -1):
            doc["_id"] = str(doc["_id"])
            # Convert datetime to string for JSON serialization
            if doc.get("upload_date"):
                doc["upload_date"] = doc["upload_date"].isoformat()
            files.append(doc)
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{file_id}")
async def get_file(file_id: str):
    """Get specific file metadata"""
    try:
        file_doc = await files_collection.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_doc["_id"] = str(file_doc["_id"])
        if file_doc.get("upload_date"):
            file_doc["upload_date"] = file_doc["upload_date"].isoformat()
        return file_doc
    except Exception as e:
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid file ID format")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete file from Cloudinary and database"""
    try:
        # Get file metadata
        file_doc = await files_collection.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete from Cloudinary
        try:
            cloudinary.uploader.destroy(file_doc["cloudinary_public_id"])
        except Exception as e:
            print(f"Warning: Failed to delete from Cloudinary: {str(e)}")
        
        # Delete associated knowledge entries
        if file_doc.get("knowledge_ids"):
            for knowledge_id in file_doc["knowledge_ids"]:
                try:
                    await knowledge_collection.delete_one({"_id": ObjectId(knowledge_id)})
                except Exception as e:
                    print(f"Warning: Failed to delete knowledge {knowledge_id}: {str(e)}")
        
        # Delete file metadata
        await files_collection.delete_one({"_id": ObjectId(file_id)})
        
        return {"message": "File deleted successfully"}
    except Exception as e:
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid file ID format")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{file_id}/download")
async def get_file_download_url(file_id: str):
    """Get Cloudinary download URL for file"""
    try:
        file_doc = await files_collection.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {"download_url": file_doc["cloudinary_url"]}
    except Exception as e:
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid file ID format")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{file_id}/signed-url")
async def get_signed_url(file_id: str):
    """Generate signed URL for secure file access"""
    try:
        file_doc = await files_collection.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Generate signed URL for secure access
        signed_url = cloudinary.utils.cloudinary_url(
            file_doc["cloudinary_public_id"],
            resource_type="raw",
            sign_url=True,
            expires_at=int((datetime.now().timestamp() + 3600))  # 1 hour expiry
        )[0]
        
        return {
            "signed_url": signed_url,
            "expires_in": 3600  # seconds
        }
    except Exception as e:
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid file ID format")
        raise HTTPException(status_code=500, detail=str(e))