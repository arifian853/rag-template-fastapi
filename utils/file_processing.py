import pandas as pd
import PyPDF2
import io
import csv
from fastapi import HTTPException
from typing import List, Dict, Any, Tuple

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