# Alzheimer AI Assistant

Alzheimer AI Assistant là ứng dụng web hỗ trợ phân tích ảnh MRI não và hỏi đáp kiến thức y khoa về bệnh Alzheimer. Dự án kết hợp mô hình Vision Transformer (ViT), OpenAI API, Retrieval-Augmented Generation (RAG), ChromaDB và Redis trong một backend FastAPI được đóng gói bằng Docker.

> Lưu ý: Ứng dụng chỉ phục vụ mục đích học thuật, nghiên cứu và minh họa kỹ thuật. Kết quả từ hệ thống không thay thế chẩn đoán hoặc tư vấn y khoa từ bác sĩ chuyên môn.

## Tính Năng Chính

- Phân tích ảnh MRI não bằng mô hình ViT đã huấn luyện.
- Phân loại ảnh thành 4 nhóm: `NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`.
- Tạo phần giải thích kết quả bằng GPT-4o Vision.
- Chat RAG dựa trên tài liệu Alzheimer đã được nhúng vào ChromaDB.
- Lưu ngữ cảnh phiên phân tích/chat bằng Redis, có fallback in-memory khi Redis không khả dụng.
- Giao diện web đơn giản bằng HTML, TailwindCSS và JavaScript thuần.
- Hỗ trợ Docker Compose local và triển khai AWS bằng ECR + ECS Fargate + Application Load Balancer.

## Kiến Trúc

```text
Browser
  |
  | HTTP
  v
FastAPI app
  |-- /                 -> giao diện web
  |-- /analyze          -> upload MRI, chạy ViT, gọi GPT-4o Vision
  |-- /chat             -> RAG chat với tài liệu Alzheimer
  |-- /health           -> liveness check
  |-- /ready            -> readiness check cho Docker/AWS
  |
  |-- Keras/TensorFlow  -> model/model_mri.keras
  |-- ChromaDB          -> chroma_db/
  |-- Redis             -> session/cache
  |-- OpenAI API        -> GPT-4o, GPT-4o mini, text-embedding-3-small
```

## Công Nghệ

- Backend: FastAPI, Uvicorn
- AI/ML: TensorFlow, Keras, Vision Transformer
- LLM/Vision: OpenAI API
- Vector database: ChromaDB
- Cache/session: Redis
- Frontend: HTML, TailwindCSS, Vanilla JavaScript
- Container: Docker, Docker Compose
- Cloud deployment: Amazon ECR, Amazon ECS Fargate, Application Load Balancer

## Cấu Trúc Thư Mục

```text
.
├── app/
│   ├── config.py
│   ├── main.py
│   ├── models/
│   ├── routes/
│   ├── schemas/
│   └── services/
├── chroma_db/
├── data/
├── model/
│   └── model_mri.keras
├── templates/
│   └── index.html
├── Dockerfile
├── docker-compose.yml
├── process_data.py
├── requirements.txt
└── run.py
```

## Biến Môi Trường

Tạo file `.env` ở thư mục gốc:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
CHROMA_PATH=./chroma_db
COLLECTION_NAME=alzheimer
VIT_MODEL_PATH=./model/model_mri.keras
IMAGE_SIZE=128
MAX_UPLOAD_MB=10
ANONYMIZED_TELEMETRY=False
```

Khi chạy bằng Docker Compose, `REDIS_URL` đã được cấu hình trong `docker-compose.yml`:

```env
REDIS_URL=redis://redis:6379/0
```

Khi chạy trực tiếp bằng Python ngoài Docker, nếu có Redis local thì dùng:

```env
REDIS_URL=redis://localhost:6379/0
```

Không commit `.env` hoặc API key thật lên Git. Nếu API key từng bị lộ, cần revoke/rotate key ngay.

## Chuẩn Bị Dữ Liệu Và Model

Ứng dụng cần 2 tài nguyên trước khi chạy:

- Model ViT: `model/model_mri.keras`
- Vector database: `chroma_db/`

Nếu cần tạo lại vector database từ dữ liệu đã parse, chạy:

```bash
python process_data.py
```

Lưu ý: `process_data.py` có thể cần thêm các thư viện xử lý PDF/text như `PyMuPDF`, `pdfplumber` và `langchain-text-splitters` tùy workflow tạo dữ liệu. Web app runtime chính dùng các package trong `requirements.txt`.

## Chạy Local Bằng Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Ứng dụng chạy tại:

```text
http://localhost:8000
```

Kiểm tra trạng thái:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Chạy Local Bằng Docker Compose

Build và chạy:

```bash
docker compose up -d --build
```

Xem trạng thái:

```bash
docker compose ps
```

Xem log:

```bash
docker compose logs -f web
docker compose logs -f redis
```

Tắt stack:

```bash
docker compose down
```

Ứng dụng chạy tại:

```text
http://localhost:8000
```

## API Endpoints

### `GET /`

Trả về giao diện web.

### `GET /health`

Liveness check. Endpoint này trả về thông tin cơ bản của app, model path, Chroma path và số session đang hoạt động.

### `GET /ready`

Readiness check cho Docker/AWS. Endpoint trả `200` khi model ViT load được và ChromaDB khả dụng. Nếu app chưa sẵn sàng, endpoint trả `503`.

### `POST /analyze`

Upload ảnh MRI và nhận kết quả phân tích.

Form data:

- `file`: ảnh MRI
- `session_id`: mã phiên, dùng để liên kết kết quả ViT với chat RAG

### `POST /chat`

Hỏi đáp với RAG.

Body:

```json
{
  "session_id": "session-id",
  "question": "Triệu chứng Alzheimer giai đoạn nhẹ là gì?"
}
```

### `POST /reset`

Xóa lịch sử chat và kết quả ViT của một session.

## Triển Khai AWS

Dự án đã được triển khai theo mô hình:

```text
Build Docker image local
  -> Push image lên Amazon ECR
  -> Chạy container bằng Amazon ECS Fargate
  -> Public qua Application Load Balancer
```

Tên mô hình triển khai:

```text
Container deployment with Amazon ECR, Amazon ECS Fargate and Application Load Balancer
```

### 1. Build Image Local

```bash
docker build -t assistant-desease-web:latest .
```

### 2. Login ECR

```powershell
$AWS_REGION="us-east-1"
$AWS_ACCOUNT_ID="YOUR_AWS_ACCOUNT_ID"
$REPO_NAME="alzheimer-assistant"

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
```

### 3. Tag Và Push Image

```powershell
docker tag assistant-desease-web:latest "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:latest"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:latest"
```

### 4. ECS Task Definition

Cấu hình chính:

- Launch type: Fargate
- Container port: `8000`
- CPU: tối thiểu `2 vCPU`
- Memory: khuyến nghị `4 GB` hoặc `8 GB`
- Image: image URI từ ECR
- Health check path phía ALB: `/ready`

Environment variables trên ECS:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
CHROMA_PATH=./chroma_db
COLLECTION_NAME=alzheimer
VIT_MODEL_PATH=./model/model_mri.keras
IMAGE_SIZE=128
MAX_UPLOAD_MB=10
```

Nếu dùng Redis managed như ElastiCache:

```env
REDIS_URL=redis://your-redis-endpoint:6379/0
```

Nếu chưa cấu hình Redis trên AWS, app vẫn có fallback in-memory, nhưng session sẽ mất khi task restart hoặc scale.

### 5. Application Load Balancer

Cấu hình:

- Listener: HTTP `80` hoặc HTTPS `443`
- Target group type: `IP`
- Target port: `8000`
- Health check path: `/ready`
- Success code: `200`

Security group:

- ALB inbound: HTTP `80` hoặc HTTPS `443` từ internet
- ECS task inbound: TCP `8000` chỉ từ security group của ALB

### 6. Redeploy Sau Khi Push Image Mới

```powershell
aws ecs update-service `
  --region us-east-1 `
  --cluster alzheimer-cluster-1 `
  --service alzheimer-service `
  --force-new-deployment
```

## Ghi Chú Vận Hành

- Không nên phụ thuộc lâu dài vào image tag `latest`; nên dùng tag version như `v1.0.0`, `v1.0.1`.
- Nên đưa `OPENAI_API_KEY` vào AWS Secrets Manager thay vì lưu plain text trong ECS environment variables.
- Nên bật HTTPS cho ALB bằng AWS Certificate Manager nếu triển khai công khai.
- Với task AI/TensorFlow, cần theo dõi memory. Nếu upload ảnh làm task restart, tăng memory lên `8 GB`.
- Nếu `/ready` fail, kiểm tra model path, ChromaDB và CloudWatch Logs.
- Nếu `/analyze` timeout, kiểm tra ALB idle timeout và thời gian gọi GPT-4o Vision.

## Debug Nhanh Trên AWS

Xem service events:

```powershell
aws ecs describe-services `
  --region us-east-1 `
  --cluster alzheimer-cluster-1 `
  --services alzheimer-service `
  --query "services[0].events[0:10].[createdAt,message]" `
  --output table
```

Xem task bị stop:

```powershell
aws ecs list-tasks `
  --region us-east-1 `
  --cluster alzheimer-cluster-1 `
  --service-name alzheimer-service `
  --desired-status STOPPED
```

Xem CloudWatch log:

```powershell
aws logs tail "/ecs/alzheimer-assistant" `
  --region us-east-1 `
  --since 1h
```

## So Sánh Với Deploy Trực Tiếp EC2

Cách đang dùng là ECS Fargate: AWS quản lý hạ tầng chạy container, tự thay task khi lỗi và tích hợp tốt với ALB.

Deploy trực tiếp EC2 là cách thuê một máy ảo, SSH vào máy, tự cài Docker/Python và tự chạy app bằng `docker compose up` hoặc `python run.py`. Cách EC2 dễ hiểu hơn khi học ban đầu, nhưng phải tự quản lý server, disk, restart, log, security và scale.

Với ứng dụng này, ECS Fargate phù hợp hơn vì app đã được container hóa, có health check `/ready`, cần restart tự động và có thể scale về sau.

## Disclaimer

Ứng dụng này không phải thiết bị y tế và không cung cấp chẩn đoán y khoa. Kết quả phân tích ảnh MRI và câu trả lời từ AI chỉ có giá trị tham khảo, phục vụ mục đích học tập, nghiên cứu và minh họa kỹ thuật.
