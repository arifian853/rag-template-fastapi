# 🤖 RAG Backend API - Knowledge Management System

Backend API untuk sistem manajemen pengetahuan dengan teknologi Retrieval-Augmented Generation (RAG) menggunakan FastAPI, MongoDB, dan Google Gemini AI.

## 📋 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Tech Stack](#-tech-stack)
- [Struktur Proyek](#-struktur-proyek)
- [Quick Start](#-quick-start)
- [Konfigurasi](#-konfigurasi)
- [API Endpoints](#-api-endpoints)
- [File Upload](#-file-upload)
- [Docker Deployment](#-docker-deployment)
- [Database Setup](#-database-setup)

## 🚀 Fitur Utama

### 📚 Knowledge Management
- **CRUD Operations**: Create, Read, Update, Delete knowledge items
- **Pagination**: Dukungan pagination dengan sorting (newest/oldest)
- **Vector Search**: Pencarian semantik menggunakan embeddings
- **Metadata Support**: Penyimpanan metadata untuk setiap knowledge item

### 🤖 AI Chat System
- **RAG Implementation**: Retrieval-Augmented Generation dengan Google Gemini
- **Context-Aware**: Chat dengan konteks dari knowledge base
- **History Support**: Dukungan chat history untuk percakapan berkelanjutan
- **Similarity Search**: Pencarian dokumen relevan berdasarkan cosine similarity

### 📁 File Processing
- **Multi-Format Support**: PDF, TXT, Excel (.xlsx), CSV
- **Batch Processing**: Upload dan proses multiple files
- **Automatic Extraction**: Ekstraksi teks otomatis dari berbagai format
- **Metadata Preservation**: Penyimpanan informasi file original

### 🔍 Vector Search
- **Sentence Transformers**: Menggunakan model `all-MiniLM-L6-v2`
- **MongoDB Vector Index**: Optimized vector search dengan MongoDB
- **Cosine Similarity**: Algoritma pencarian berdasarkan similarity score
- **Real-time Embeddings**: Generate embeddings secara real-time

## 🛠 Tech Stack

- **Framework**: FastAPI 
- **Database**: MongoDB dengan Motor (async driver)
- **AI Model**: Google Gemini 2.5 Flash
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **File Processing**: PyPDF2, pandas, openpyxl
- **Deployment**: Docker & Docker Compose
- **Environment**: Python 3.11+

## 📁 Struktur Proyek

```
rag/
├── app.py                 # Main FastAPI application
├── create_index.py        # MongoDB vector index setup
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
└── .dockerignore        # Docker ignore rules
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- MongoDB Atlas account atau MongoDB local
- Google AI API Key

### 1. Clone & Setup
```bash
cd d:/chatbot_cs/rag
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
cp .env.example .env
# Edit .env dengan credentials Anda
```

### 3. Database Setup
```bash
python create_index.py
```

### 4. Run Development Server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 7860
```

API akan tersedia di: `http://localhost:7860`

## ⚙️ Konfigurasi

### Environment Variables (.env)
```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/chatbot_cs
GOOGLE_API_KEY=your_google_ai_api_key_here
```

### MongoDB Setup
- Database: `chatbot_cs` (Silakan sesuaikan dengan nama database yang digunakan)
- Collection: `rag_data_knowledge` (Silakan sesuaikan dengan nama collection yang digunakan)
- Index: Vector search index untuk embeddings

## 📡 API Endpoints

### Knowledge Management

#### GET `/knowledge`
List knowledge items dengan pagination
```json
{
  "page": 1,
  "limit": 15,
  "sort_order": "newest"
}
```

#### POST `/add-knowledge`
Tambah knowledge item baru
```json
{
  "title": "Judul Knowledge",
  "content": "Konten knowledge...",
  "source": "manual",
  "metadata": {}
}
```

#### GET `/knowledge/{knowledge_id}`
Get knowledge item by ID

#### PUT `/knowledge/{knowledge_id}`
Update knowledge item
```json
{
  "title": "Updated Title",
  "content": "Updated content...",
  "metadata": {}
}
```

#### DELETE `/knowledge/{knowledge_id}`
Delete knowledge item

### Chat System

#### POST `/chat`
Chat dengan AI menggunakan RAG
```json
{
  "message": "Pertanyaan Anda",
  "history": [
    {
      "role": "user",
      "content": "Pertanyaan sebelumnya"
    },
    {
      "role": "assistant", 
      "content": "Jawaban AI"
    }
  ]
}
```

### File Upload

#### POST `/upload-file`
Upload dan proses file (PDF, TXT, Excel, CSV)
```bash
curl -X POST "http://localhost:7860/upload-file" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.pdf"
```

## 📁 File Upload

### Supported Formats

#### PDF Files (.pdf)
- Ekstraksi teks otomatis
- Preservasi metadata PDF
- Support untuk PDF dengan teks dan gambar

#### Text Files (.txt)
- Import langsung konten teks
- Encoding UTF-8 support

#### Excel Files (.xlsx)
- Proses per baris sebagai knowledge item
- Deteksi header otomatis
- Support multiple sheets

#### CSV Files (.csv)
- Import dengan header detection
- Flexible column mapping
- UTF-8 encoding support

## 🐳 Docker Deployment

### Build & Run
```bash
# Build image
docker build -t rag-backend .

# Run container
docker run -p 7860:7860 --env-file .env rag-backend
```

### Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment
```bash
# Build for production
docker-compose -f docker-compose.yml up -d --build

# Health check
curl http://localhost:7860/knowledge
```

## 🗄️ Database Setup

### MongoDB Atlas (Recommended)
1. Buat cluster di MongoDB Atlas
2. Whitelist IP address
3. Buat database user
4. Copy connection string ke `.env`

### Local MongoDB
```bash
# Install MongoDB
# Start MongoDB service
mongod --dbpath /path/to/data

# Update .env
MONGODB_URI=mongodb://localhost:27017/chatbot_cs
```

### Vector Index Setup
```bash
python create_index.py
```
