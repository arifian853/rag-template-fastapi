from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import knowledge_routes, chat_routes, system_prompt_routes, file_routes, auth_route
import uvicorn

app = FastAPI(title="RAG Knowledge Management API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://cheap-rag.netlify.app"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with proper prefixes
app.include_router(auth_route.router, prefix="/auth", tags=["auth"])
app.include_router(knowledge_routes.router, tags=["knowledge"])
app.include_router(chat_routes.router, tags=["chat"])
app.include_router(system_prompt_routes.router, prefix="/system-prompts", tags=["system-prompts"])
app.include_router(file_routes.router, prefix="/files", tags=["files"])

@app.get("/")
async def root():
    return {"message": "RAG Knowledge Management API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)