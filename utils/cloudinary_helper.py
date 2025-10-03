import io
import base64
import cloudinary
import cloudinary.uploader
from datetime import datetime
from fastapi import HTTPException

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