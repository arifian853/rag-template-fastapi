from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import routes
from routes.chat_routes import router as chat_router
from routes.knowledge_routes import router as knowledge_router
from routes.system_prompt_routes import router as system_prompt_router
from routes.file_routes import router as file_router

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, tags=["Chat"])
app.include_router(knowledge_router, tags=["Knowledge"])
app.include_router(system_prompt_router, prefix="/system-prompts", tags=["System Prompts"])
app.include_router(file_router, prefix="/files", tags=["Files"])

# Add backward compatibility endpoints at root level
from routes.file_routes import upload_file_legacy, upload_csv_custom, upload_excel_custom

app.post("/upload-file", tags=["Files - Legacy"])(upload_file_legacy)
app.post("/upload-csv-custom", tags=["Files - Legacy"])(upload_csv_custom)
app.post("/upload-excel-custom", tags=["Files - Legacy"])(upload_excel_custom)

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)