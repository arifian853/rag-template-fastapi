from config import files_collection
from datetime import datetime
from typing import List
from fastapi import HTTPException

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