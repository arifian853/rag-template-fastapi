import os
import certifi
import motor.motor_asyncio as motor
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import cloudinary
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.environ["MONGODB_URI"] 
DB_NAME = os.getenv("MONGODB_DB", "chatbot_cs")

client = motor.AsyncIOMotorClient(
    MONGODB_URI,
    server_api=ServerApi("1"),
    tls=True,
    tlsCAFile=certifi.where(),
    tlsDisableOCSPEndpointCheck=True, 
    serverSelectionTimeoutMS=30000,
    retryReads=True,
    retryWrites=True,
    appname="rag-fastapi",
)

db = client[DB_NAME]
knowledge_collection = db.get_collection("rag_data_knowledge")
system_prompt_collection = db.get_collection("system_prompts")
files_collection = db.get_collection("files")
users_collection = db.get_collection("users")

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

model = SentenceTransformer("all-MiniLM-L6-v2")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

async def get_database():
    return db
