from fastapi import APIRouter, HTTPException
from models import KnowledgeItem, UpdateKnowledgeItem, PaginatedResponse
from services.knowledge_service import add_knowledge
from config import knowledge_collection
from utils.embedding import generate_embedding
from bson import ObjectId
from typing import List, Dict, Any

router = APIRouter()

@router.get("/knowledge", response_model=PaginatedResponse)
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

@router.get("/knowledge/all", response_model=List[Dict[str, Any]])
async def list_all_knowledge():
    """List all knowledge items (for backward compatibility)"""
    items = []
    async for doc in knowledge_collection.find({}, {"embedding": 0}):
        doc["_id"] = str(doc["_id"])
        items.append(doc)
    return items

@router.post("/add-knowledge")
async def create_knowledge(knowledge: KnowledgeItem):
    """Add a new knowledge item"""
    result = await add_knowledge(knowledge)
    return {"id": str(result), "message": "Knowledge added successfully"}

@router.get("/knowledge/{knowledge_id}")
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

@router.put("/knowledge/{knowledge_id}")
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

@router.delete("/knowledge/{knowledge_id}")
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

@router.delete("/knowledge")
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