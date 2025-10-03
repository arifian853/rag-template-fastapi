import os
import motor.motor_asyncio
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import cloudinary
import cloudinary.uploader
import cloudinary.api
import cloudinary.utils
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI, server_api=ServerApi('1'))
db = client.get_database("chatbot_cs")
knowledge_collection = db.get_collection("rag_data_knowledge")
system_prompt_collection = db.get_collection("system_prompts")
files_collection = db.get_collection("files")

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))