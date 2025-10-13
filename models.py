from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = []

class KnowledgeItem(BaseModel):
    title: str
    content: str
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class UpdateKnowledgeItem(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SystemPrompt(BaseModel):
    name: str
    prompt: str
    is_active: bool = False
    is_default: bool = False
    description: Optional[str] = None

class UpdateSystemPrompt(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None
    is_active: Optional[bool] = None
    is_default: Optional[bool] = None
    description: Optional[str] = None

class FileMetadata(BaseModel):
    filename: str
    original_filename: str
    cloudinary_url: str
    cloudinary_public_id: str
    file_type: str  # pdf, excel, csv
    file_size: int
    upload_date: Optional[datetime] = None
    knowledge_ids: List[str] = []
    metadata: Optional[Dict[str, Any]] = None

# User Authentication Models
class User(BaseModel):
    username: str
    password: str
    created_at: Optional[datetime] = None
    is_active: bool = True

class UserCreate(BaseModel):
    username: str
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None

class UserResponse(BaseModel):
    id: str
    username: str
    created_at: datetime
    is_active: bool

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class TokenData(BaseModel):
    username: Optional[str] = None

class PaginationParams(BaseModel):
    page: int = 1
    limit: int = 15
    sort_order: str = "newest"  # newest or oldest

class PaginatedResponse(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_prev: bool