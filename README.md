<div align="center">
  <h1>Alzheimer AI Assistant</h1>
  <p><i>Hệ thống phát hiện sớm bệnh Sa sút trí tuệ (Alzheimer) qua ảnh MRI và Trợ lý AI Hỏi đáp Y khoa (RAG)</i></p>
  
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://tensorflow.org/)
  [![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
  [![OpenAI](https://img.shields.io/badge/OpenAI-412991.svg?style=for-the-badge&logo=OpenAI&logoColor=white)](https://openai.com/)
</div>

---

## Về Dự Án

**Alzheimer AI Assistant** là một ứng dụng y khoa, kết hợp thị giác máy tính (Computer Vision) và xử lý ngôn ngữ tự nhiên (NLP), được xây dựng với mục tiêu hỗ trợ chẩn đoán và cung cấp kiến thức về bệnh Alzheimer.

Dự án có 2 luồng tính năng chính:
1. **Phân tích ảnh MRI**: Sử dụng mô hình **Vision Transformer (ViT)** để phân loại mức độ sa sút trí tuệ thành 4 cấp độ (NonDemented, VeryMildDemented, MildDemented, ModerateDemented). Sau đó, ảnh được đưa qua **GPT-4o Vision** để cung cấp thêm phân tích chi tiết.
2. **Trợ lý Ảo Y Khoa (RAG Chat)**: Hệ thống Retrieval-Augmented Generation (RAG) sử dụng cơ sở dữ liệu vector ChromaDB để trích xuất các tài liệu y khoa chuẩn xác, từ đó trả lời các câu hỏi của người dùng.

---

## Công Nghệ Sử Dụng

- **Backend Framework**: FastAPI
- **Deep Learning**: TensorFlow / Keras
- **LLM & Vision**: OpenAI API (gpt-4o, gpt-4o-mini, text-embedding-3)
- **Vector Database**: ChromaDB
- **Session/Cache**: Redis
- **Frontend**: HTML/Vanilla JS kết hợp TailwindCSS
- **Deployment**: Docker & Docker Compose

---

## Hướng Dẫn Chạy

### Bước 1: Cấu hình Môi trường (.env)
Tạo file `.env` tại thư mục gốc của dự án và điền thông tin sau:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxx
CHROMA_PATH=./chroma_db
COLLECTION_NAME=alzheimer
VIT_MODEL_PATH=./model/model_mri.keras
IMAGE_SIZE=128
REDIS_URL=redis://localhost:6379/0
```

### Bước 2: Chuẩn bị Dữ liệu (Vector DB)
Chạy script sau để nạp tài liệu vào DB:

```bash
python process_data.py
```

### Bước 3: Chuẩn Bị Model ViT
Đảm bảo bạn đã tải file model của mình (`model_mri.keras`) và đặt vào thư mục `./model/`.

### Bước 4: Khởi động Ứng Dụng

**Cách 1: Chạy bằng Docker (Khuyên dùng)**
```bash
docker-compose up -d --build
```

**Cách 2: Chạy qua Python**
Đảm bảo đã cài thư viện trong `requirements.txt` và bật Redis Server ở Port 6379:
```bash
python run.py
```

Ứng dụng sẽ khả dụng tại: **http://localhost:8000**
