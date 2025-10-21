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
import ssl

# Load environment variables
load_dotenv()

# MongoDB connection with proper SSL and connection settings
MONGODB_URI = os.getenv("MONGODB_URI")

# Create SSL context for MongoDB Atlas
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Motor client with optimized settings
client = motor.motor_asyncio.AsyncIOMotorClient(
    MONGODB_URI, 
    server_api=ServerApi('1'),
    # SSL Configuration
    tls=True,
    tlsAllowInvalidCertificates=True,
    tlsAllowInvalidHostnames=True,
    # Connection Pool Settings
    maxPoolSize=50,
    minPoolSize=5,
    maxIdleTimeMS=30000,
    # Timeout Settings
    serverSelectionTimeoutMS=5000,
    socketTimeoutMS=20000,
    connectTimeoutMS=20000,
    # Retry Settings
    retryWrites=True,
    retryReads=True,
    # Heartbeat
    heartbeatFrequencyMS=10000
)

db = client.get_database("chatbot_cs")
knowledge_collection = db.get_collection("rag_data_knowledge")
system_prompt_collection = db.get_collection("system_prompts")
files_collection = db.get_collection("files")
users_collection = db.get_collection("users")

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

# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Function untuk compatibility dengan auth routes
async def get_database():
    return db