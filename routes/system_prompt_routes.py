from fastapi import APIRouter, HTTPException, Depends
from models import SystemPrompt, UpdateSystemPrompt
from services.system_prompt_service import get_active_system_prompt, initialize_default_prompt, ensure_default_prompt_exists
from config import system_prompt_collection
from bson import ObjectId
import asyncio
from .auth_route import get_current_user

router = APIRouter()

@router.get("/")
async def list_system_prompts(current_user: dict = Depends(get_current_user)):
    """List all system prompts"""
    prompts = []
    async for doc in system_prompt_collection.find({}):
        doc["_id"] = str(doc["_id"])
        prompts.append(doc)
    return prompts

@router.get("/active")
async def get_active_prompt(current_user: dict = Depends(get_current_user)):
    """Get the currently active system prompt"""
    prompt_doc = await system_prompt_collection.find_one({"is_active": True})
    if not prompt_doc:
        raise HTTPException(status_code=404, detail="No active system prompt found")
    
    prompt_doc["_id"] = str(prompt_doc["_id"])
    return prompt_doc

@router.post("/")
async def create_system_prompt(prompt: SystemPrompt, current_user: dict = Depends(get_current_user)):
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

@router.put("/{prompt_id}")
async def update_system_prompt(prompt_id: str, prompt_data: UpdateSystemPrompt, current_user: dict = Depends(get_current_user)):
    """Update a system prompt"""
    try:
        update_data = {}
        
        if prompt_data.name is not None:
            update_data["name"] = prompt_data.name
        if prompt_data.prompt is not None:
            update_data["prompt"] = prompt_data.prompt
        if prompt_data.description is not None:
            update_data["description"] = prompt_data.description
        if prompt_data.is_active is not None:
            update_data["is_active"] = prompt_data.is_active
            
            # If setting this as active, deactivate all others
            if prompt_data.is_active:
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

@router.delete("/{prompt_id}")
async def delete_system_prompt(prompt_id: str, current_user: dict = Depends(get_current_user)):
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
            # Ensure default prompt exists and activate it
            default_prompt = await ensure_default_prompt_exists()
            
            # Activate the default prompt
            await system_prompt_collection.update_one(
                {"_id": default_prompt["_id"]},
                {"$set": {"is_active": True}}
            )
        
        return {"message": "System prompt deleted successfully"}
    except Exception as e:
        if "ObjectId" in str(e):
            raise HTTPException(status_code=400, detail="Invalid ID format")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{prompt_id}/activate")
async def activate_system_prompt(prompt_id: str, current_user: dict = Depends(get_current_user)):
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

@router.post("/initialize")
async def initialize_default_prompt_endpoint():
    """Initialize default system prompt if none exists"""
    try:
        result = await initialize_default_prompt()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset-to-default")
async def reset_to_default(current_user: dict = Depends(get_current_user)):
    """Reset system to use default prompt - auto-create if not exists"""
    try:
        # Ensure default prompt exists (create if not)
        default_prompt = await ensure_default_prompt_exists()
        
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
        
        return {
            "message": "Reset to default prompt successfully",
            "default_prompt": {
                "id": str(default_prompt["_id"]),
                "name": default_prompt["name"],
                "description": default_prompt.get("description", "")
            }
        }
    except Exception as e:
        print(f"Error in reset_to_default: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))