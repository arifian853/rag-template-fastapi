import google.generativeai as genai
import asyncio
from typing import List, Dict
from services.knowledge_service import search_similar_documents
from services.system_prompt_service import get_active_system_prompt
from config import files_collection
from bson import ObjectId

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
    
    # Enhance sources with file information
    enhanced_sources = []
    for doc in relevant_docs:
        source_data = {
            "_id": str(doc["_id"]),
            "title": doc["title"], 
            "content": doc["content"], 
            "source": doc["source"],
            "similarity_score": round(doc["score"], 4),
            "file_info": None
        }
        
        # Check if this knowledge has associated file
        if doc.get("source") and doc.get("metadata", {}).get("filename"):
            # Try to find file metadata
            filename = doc["metadata"]["filename"]
            file_doc = await files_collection.find_one({"original_filename": filename})
            
            if file_doc:
                source_data["file_info"] = {
                    "file_id": str(file_doc["_id"]),
                    "filename": file_doc["original_filename"],
                    "file_type": file_doc["file_type"],
                    "cloudinary_url": file_doc["cloudinary_url"],
                    "upload_date": file_doc.get("upload_date").isoformat() if file_doc.get("upload_date") else None
                }
        
        enhanced_sources.append(source_data)
    
    return {
        "response": response.text,
        "sources": enhanced_sources
    }