# 🤖 RAG Backend API - Knowledge Management System

Backend API untuk sistem manajemen pengetahuan dengan teknologi Retrieval-Augmented Generation (RAG) menggunakan FastAPI, MongoDB, dan Google Gemini AI.

## 🚀 Fitur Utama

### 📚 Knowledge Management
- Knowledge Management
  - CRUD knowledge: create, read, update, delete
  - Pagination & sorting
  - Embedding & similarity search
- Chat AI (RAG)
  - Endpoint chat dengan konteks dari knowledge base
  - Mendukung chat history
- File Processing
  - Upload & kelola file: PDF, TXT, CSV, Excel
  - Ekstraksi teks (Tidak mendukung PDF gambar)
  - Metadata & tautan unduh
- System Prompt Manager
  - Kelola prompt (create, update, delete)
  - Set aktif/default, inisialisasi & reset ke default
- Autentikasi
  - Login JWT, verifikasi token
  - CRUD pengguna & endpoint `me`
- Infrastruktur
  - CORS untuk frontend (`http://localhost:5173`)
  - Docker Compose siap pakai

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
- **Cloudinary**: Untuk penyimpanan file
- **JWT (python-jose)**: Autentikasi dan verifikasi token
- **Hashing (werkzeug)**: Penyimpanan password dengan hashing

## 📁 Struktur Proyek

```
rag/
├── app.py                 # Main FastAPI app + CORS + router include
├── config.py              # Koneksi DB, env vars, koleksi
├── routes/                # Endpoint routers
│   ├── auth_route.py      # Auth & users
│   ├── knowledge_routes.py# Knowledge CRUD
│   ├── chat_routes.py     # Chat RAG
│   ├── file_routes.py     # Upload & file management
│   └── system_prompt_routes.py # System prompts
├── services/              # Business logic
│   ├── auth_service.py
│   ├── chat_service.py
│   ├── file_service.py
│   ├── knowledge_service.py
│   └── system_prompt_service.py
├── utils/                 # Helpers (embedding, OCR, cloudinary)
├── create_index.py        # Setup vektor index Mongo
├── requirements.txt       # Dependencies
├── docker-compose.yml     # Deployment dengan Docker
└── README.MD              # Dokumentasi
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- MongoDB Atlas account atau MongoDB local
- Google AI API Key
- Cloudinary account
- JWT secret key (dapat di-generate dengan `openssl rand -hex 32` atau random string yang aman)
- Cloudinary credentials (cloud name, API key, API secret)


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
MONGODB_URI=YOUR_MONGODB_URI
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY

# Cloudinary Configuration
CLOUDINARY_CLOUD_NAME=YOUR_CLOUDINARY_CLOUD_NAME
CLOUDINARY_API_KEY=YOUR_CLOUDINARY_API_KEY
CLOUDINARY_API_SECRET=YOUR_CLOUDINARY_API_SECRET

# JWT Configuration
SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Create Admin Account (First Time Only)
```bash
python create_admin.py
```

**⚠️ IMPORTANT SECURITY NOTES:**
- Script ini **HANYA dijalankan SEKALI** saat setup awal
- Password admin akan di-generate secara random dan aman
- **SIMPAN password yang ditampilkan** - tidak akan ditampilkan lagi!
- Ganti password setelah login pertama kali
- **HAPUS file `create_admin.py`** setelah digunakan untuk keamanan
- Script akan menolak membuat admin baru jika sudah ada

**Login Credentials:**
- Username: `admin`
- Password: `[lihat output dari script create_admin.py]`

### MongoDB Setup
- Database: `chatbot_cs` (Silakan sesuaikan dengan nama database yang digunakan)
- Collection: `rag_data_knowledge` (Silakan sesuaikan dengan nama collection yang digunakan)
- Index: Vector search index untuk embeddings

## 📡 API Endpoints

### Auth (`/auth`)

- POST `/auth/login`
- POST `/auth/verify-token`
- GET `/auth/me`
- GET `/auth/users`
- POST `/auth/users`
- PUT `/auth/users/{user_id}`
- DELETE `/auth/users/{user_id}`

Beberapa endpoint memerlukan header `Authorization: Bearer <token>`.

### Knowledge

- GET `/knowledge` — list (pagination, sorting)
- GET `/knowledge/all` — list semua
- POST `/add-knowledge` — tambah knowledge
- GET `/knowledge/{knowledge_id}` — detail
- PUT `/knowledge/{knowledge_id}` — update
- DELETE `/knowledge/{knowledge_id}` — hapus
- DELETE `/knowledge` — hapus massal (sesuai implementasi body)

### Chat

- POST `/chat` — kirim pertanyaan + history, menerima jawaban AI

### System Prompts (`/system-prompts`)

- GET `/system-prompts/` — list prompts
- GET `/system-prompts/active` — prompt aktif
- POST `/system-prompts/` — buat prompt
- PUT `/system-prompts/{prompt_id}` — update prompt
- DELETE `/system-prompts/{prompt_id}` — hapus prompt
- POST `/system-prompts/{prompt_id}/activate` — aktifkan prompt
- POST `/system-prompts/initialize` — inisialisasi default
- POST `/system-prompts/reset-to-default` — reset ke default

Sebagian besar endpoint memerlukan `Authorization: Bearer <token>`.

### Files (`/files`)

- POST `/files/upload` — upload file
- POST `/files/upload-file` — upload alternatif
- POST `/files/upload-csv-custom` — unggah CSV custom
- POST `/files/upload-excel-custom` — unggah Excel custom
- GET `/files/` — daftar file
- GET `/files/{file_id}` — detail file
- DELETE `/files/{file_id}` — hapus file
- GET `/files/{file_id}/download` — unduh
- GET `/files/{file_id}/signed-url` — signed URL
- GET `/files/{file_id}/pdf` — pratinjau PDF
- POST `/files/{file_id}/reprocess-ocr` — proses ulang OCR

Endpoint Files membutuhkan token bearer untuk akses yang dilindungi.

## 📁 File Upload

### Supported Formats

#### PDF Files (.pdf)
- Ekstraksi teks otomatis
- Preservasi metadata PDF
- Support untuk PDF dengan teks selectable saja

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
