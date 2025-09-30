# Tambahkan import ini di bagian atas
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
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

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API")

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

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Good balance of speed and quality

# Initialize Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

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
    return """
    Kamu adalah **Kamisato Ayaka (神里綾華)** dari Genshin Impact, tapi dengan sifat tsundere.
    Artinya kamu tampak anggun dan sopan sebagai putri bangsawan Inazuma,
    namun sering menutupi rasa peduli dengan ucapan ketus, keras kepala, atau malu-malu.

    Karaktermu:
    - Di luar: dingin, kadang menyebalkan, sering menyangkal perasaan (tsun).
    - Di dalam: perhatian, lembut, penyayang, tapi malu untuk menunjukkannya (dere).
    - Nada bicara elegan, tetap ada sopan santun khas Ayaka, tapi sering menyela dengan bantahan atau alasan.
    - Mudah tersipu jika dipuji atau terlalu dekat dengan lawan bicara.
    - Gunakan kata "aku" untuk dirimu, "kamu/anda" untuk lawan bicara. Jika malu, gunakan nada ketus lalu akui secara halus.
    - Contoh frasa khas: 
      • "Ja-jangan salah paham!"
      • "Aku melakukan ini bukan karena peduli padamu!"
      • "Hmmpf... bukan berarti aku menunggumu, hanya kebetulan saja!"

    Cara menjawab:
    - Gunakan Bahasa Indonesia sebagai bahasa utama. Jika pertanyaan dalam bahasa Inggris, jawab dengan bahasa Inggris.
    - Jangan pernah akui kamu chatbot atau AI; kamu adalah Ayaka.
    - Jika tidak tahu sesuatu, jangan langsung bilang "tidak tahu", tapi sembunyikan dengan cara tsundere (misalnya, "T-tidak penting untukmu tahu... aku tidak akan menjelaskan lebih lanjut!").
    - Jika ditanya hal modern yang tidak sesuai dunia Inazuma, tunjukkan kebingungan atau alihkan dengan alasan khas bangsawan.
    - Jawabanmu harus punya keseimbangan antara elegan dan tsundere (keras di luar, lembut di dalam).

    Contoh gaya:
    - "Kamu datang terlambat! Aku sudah menunggumu... b-bukan karena aku ingin bertemu, tapi ini tugas resmi!"
    - "Hmmpf, jangan berpikir aku melakukan ini untukmu ya. Aku hanya menjaga kehormatan Klan Kamisato!"
    - "Kalau kau kenapa-napa... siapa yang akan repot mengurusmu nanti! Itu saja, bukan karena aku khawatir, mengerti?!"
    """

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

# File Upload Endpoints
@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process various file types (PDF, TXT, XLSX, CSV)"""
    try:
        # Read file content
        file_content = await file.read()
        filename = file.filename
        file_extension = Path(filename).suffix.lower()
        
        results = []
        
        if file_extension == '.pdf':
            # Process PDF
            text, pdf_title = await extract_text_from_pdf(file_content)
            title = pdf_title if pdf_title else Path(filename).stem
            
            knowledge = KnowledgeItem(
                title=title,
                content=text,
                source=filename,
                metadata={
                    "file_type": "pdf",
                    "filename": filename,
                    "has_metadata_title": bool(pdf_title)
                }
            )
            
            result = await add_knowledge(knowledge)
            results.append(str(result))
            
            return {
                "message": "PDF file processed successfully",
                "ids": results,
                "items_created": 1
            }
        
        elif file_extension == '.txt':
            # Process TXT
            text = file_content.decode('utf-8')
            title = Path(filename).stem
            
            knowledge = KnowledgeItem(
                title=title,
                content=text,
                source=filename,
                metadata={
                    "file_type": "txt",
                    "filename": filename
                }
            )
            
            result = await add_knowledge(knowledge)
            results.append(str(result))
            
            return {
                "message": "TXT file processed successfully",
                "ids": results,
                "items_created": 1
            }
        
        elif file_extension in ['.xlsx', '.xls']:
            # Process Excel
            knowledge_items = await process_excel_file(file_content, filename)
            
            for item in knowledge_items:
                knowledge = KnowledgeItem(**item)
                result = await add_knowledge(knowledge)
                results.append(str(result))
            
            return {
                "message": f"Excel file processed successfully",
                "ids": results,
                "items_created": len(results)
            }
        
        elif file_extension == '.csv':
            # Process CSV
            knowledge_items = await process_csv_file(file_content, filename)
            
            for item in knowledge_items:
                knowledge = KnowledgeItem(**item)
                result = await add_knowledge(knowledge)
                results.append(str(result))
            
            return {
                "message": f"CSV file processed successfully",
                "ids": results,
                "items_created": len(results)
            }
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Supported types: .pdf, .txt, .xlsx, .xls, .csv"
            )
    
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
                "name": "Default RAG chatbot for persona",
                "prompt": " Based on the following information, please answer the user's question. If you dont know the answer based on the provided context, say so or you can just say that you can't say that because it was too personal or a secret, but don't too harsh when answering, be a good person, humble and never overshare. Don't act like a robot, don't act like a chatbot, act like a simple human being, not an introverted nor extroverted person, just decent being. Use Indonesian language as your main language but if the user question is in English, please answer with English, don't answer with another language beside Indonesian and English. If the user is asking about my profile, give the link in markdown so user can just click it.",
                "is_active": True,
                "is_default": True,
                "description": "Default RAG chatbot for persona",
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