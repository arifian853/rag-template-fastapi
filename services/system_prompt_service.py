from config import system_prompt_collection
import asyncio

async def get_active_system_prompt():
    """Get the currently active system prompt"""
    prompt_doc = await system_prompt_collection.find_one({"is_active": True})
    if prompt_doc:
        return prompt_doc["prompt"]
    
    # If no active prompt, try to find default and activate it
    default_prompt = await system_prompt_collection.find_one({"is_default": True})
    if default_prompt:
        # Activate the default prompt
        await system_prompt_collection.update_one(
            {"_id": default_prompt["_id"]},
            {"$set": {"is_active": True}}
        )
        return default_prompt["prompt"]
    
    # If no default prompt exists, create one
    await initialize_default_prompt()
    
    # Get the newly created default prompt
    new_default = await system_prompt_collection.find_one({"is_default": True})
    if new_default:
        return new_default["prompt"]
    
    # Fallback to hardcoded default
    return "Based on the following information, please answer the user's question. If you don't know the answer based on the provided context, say so. Don't act like a robot, don't act like a chatbot, act like a simple human being, not an introverted nor extroverted person, just decent being. Use Indonesian language as your main language but if the user question is in English, please answer with English, don't answer with another language beside Indonesian and English."

async def initialize_default_prompt():
    """Initialize default system prompt if none exists"""
    try:
        # Check if default prompt exists
        default_exists = await system_prompt_collection.find_one({"is_default": True})
        
        if not default_exists:
            # Create default prompt regardless of other prompts existing
            default_prompt = {
                "name": "Default RAG chatbot",
                "prompt": "Based on the following information, please answer the user's question. If you don't know the answer based on the provided context, say so. Don't act like a robot, don't act like a chatbot, act like a simple human being, not an introverted nor extroverted person, just decent being. Use Indonesian language as your main language but if the user question is in English, please answer with English, don't answer with another language beside Indonesian and English.",
                "is_active": True,
                "is_default": True,
                "description": "Default RAG chatbot system prompt",
                "created_at": asyncio.get_event_loop().time()
            }
            
            # Deactivate all other prompts first
            await system_prompt_collection.update_many(
                {"is_active": True},
                {"$set": {"is_active": False}}
            )
            
            result = await system_prompt_collection.insert_one(default_prompt)
            return {"message": "Default system prompt initialized", "id": str(result.inserted_id)}
        else:
            return {"message": "Default system prompt already exists"}
    except Exception as e:
        raise Exception(str(e))

async def ensure_default_prompt_exists():
    """Ensure default prompt exists, create if not"""
    try:
        default_prompt = await system_prompt_collection.find_one({"is_default": True})
        
        if not default_prompt:
            # Create default prompt
            default_prompt_data = {
                "name": "Default RAG chatbot",
                "prompt": "Based on the following information, please answer the user's question. If you don't know the answer based on the provided context, say so. Don't act like a robot, don't act like a chatbot, act like a simple human being, not an introverted nor extroverted person, just decent being. Use Indonesian language as your main language but if the user question is in English, please answer with English, don't answer with another language beside Indonesian and English.",
                "is_active": False,  # Don't activate automatically
                "is_default": True,
                "description": "Default RAG chatbot system prompt",
                "created_at": asyncio.get_event_loop().time()
            }
            
            result = await system_prompt_collection.insert_one(default_prompt_data)
            
            # Return the newly created prompt
            return await system_prompt_collection.find_one({"_id": result.inserted_id})
        
        return default_prompt
    except Exception as e:
        raise Exception(str(e))