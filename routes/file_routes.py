from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Response, Depends
from models import KnowledgeItem
from services.knowledge_service import add_knowledge
from services.file_service import save_file_metadata
from utils.cloudinary_helper import upload_file_to_cloudinary
from config import files_collection, knowledge_collection
from bson import ObjectId
import pandas as pd
import PyPDF2
import io
import csv
import base64
import requests
import cloudinary
import cloudinary.uploader
from datetime import datetime
from utils.embedding import generate_embedding
from .auth_route import get_current_user

router = APIRouter()

# File upload endpoints (dapat diakses dengan atau tanpa prefix)
@router.post("/upload")
async def upload_file(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Upload file, save to Cloudinary, and create knowledge entries (text-based PDFs only)"""
    try:
        file_content = await file.read()
        file_extension = file.filename.split('.')[-1].lower()
        
        # Upload to Cloudinary first
        cloudinary_data = await upload_file_to_cloudinary(file_content, file.filename, file_extension)
        
        # Process file content for knowledge creation
        knowledge_ids = []
        
        if file_extension == 'pdf':
            # PDF processing - text-based only
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"
            
            # Check if PDF has extractable text
            meaningful_text = full_text.strip().replace('\n', ' ').replace(' ', '')
            
            if len(meaningful_text) < 50:
                raise HTTPException(
                    status_code=400, 
                    detail="This PDF appears to be scanned or image-based. Only text-based PDFs are supported. Please use a PDF with selectable text content."
                )
            
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
                    "processing_method": "standard",
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
        
        return {
            "message": f"File uploaded successfully to Cloudinary and processed for knowledge",
            "file_id": file_id,
            "cloudinary_url": cloudinary_data["url"],
            "knowledge_ids": knowledge_ids,
            "items_created": len(knowledge_ids)
        }
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-csv-custom")
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

@router.post("/upload-excel-custom")
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

@router.get("/")
async def list_files(current_user: dict = Depends(get_current_user)):
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

@router.get("/{file_id}")
async def get_file(file_id: str, current_user: dict = Depends(get_current_user)):
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

@router.delete("/{file_id}")
async def delete_file(file_id: str, current_user: dict = Depends(get_current_user)):
    """Delete file from Cloudinary and database"""
    try:
        # Get file metadata
        file_doc = await files_collection.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete from Cloudinary
        try:
            # Use the correct public_id to delete from Cloudinary
            public_id = file_doc["cloudinary_public_id"]
            result = cloudinary.uploader.destroy(public_id)
            print(f"Cloudinary deletion result: {result}")
            
            if result.get("result") != "ok":
                print(f"Warning: Cloudinary deletion may have failed: {result}")
                # Try with resource_type for raw files
                result = cloudinary.uploader.destroy(public_id, resource_type="raw")
                print(f"Cloudinary deletion result (raw): {result}")
        except Exception as e:
            print(f"Warning: Failed to delete from Cloudinary: {str(e)}")
        
        # DO NOT delete associated knowledge entries - keep them separate
        # Instead, remove ALL file references from knowledge entries
        if file_doc.get("knowledge_ids"):
            for knowledge_id in file_doc["knowledge_ids"]:
                try:
                    # Remove all file-related metadata
                    await knowledge_collection.update_one(
                        {"_id": ObjectId(knowledge_id)},
                        {"$unset": {
                            "metadata.filename": "",
                            "metadata.file_id": "",
                            "metadata.cloudinary_url": "",
                            "metadata.file_type": "",
                            "metadata.storage_type": ""
                        }}
                    )
                except Exception as e:
                    print(f"Warning: Failed to update knowledge {knowledge_id}: {str(e)}")
        
        # Delete file metadata from database
        await files_collection.delete_one({"_id": ObjectId(file_id)})
        
        return {"message": "File deleted successfully"}
    except Exception as e:
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid file ID format")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{file_id}/download")
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

@router.get("/{file_id}/signed-url")
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

@router.get("/{file_id}/pdf")
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

@router.get("/{file_id}/download")
async def download_file(file_id: str):
    """Download file - for PDF it converts from base64, for others redirect to Cloudinary"""
    try:
        file_doc = await files_collection.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if it's a PDF stored as base64
        if (file_doc.get("file_type") == "pdf" and 
            file_doc.get("metadata", {}).get("storage_type") == "pdf_as_base64"):
            
            # Fetch base64 content from Cloudinary
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
                                "Content-Disposition": f"attachment; filename={file_doc['original_filename']}",
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
            # For normal files, redirect to Cloudinary URL for download
            return {"download_url": file_doc["cloudinary_url"]}
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Download file error: {str(e)}")
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid file ID format")
        raise HTTPException(status_code=500, detail=str(e))