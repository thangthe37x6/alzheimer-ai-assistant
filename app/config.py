import os
import openai
import chromadb
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
CHROMA_PATH     = os.getenv("CHROMA_PATH",     "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "alzheimer")
VIT_MODEL_PATH  = os.getenv("VIT_MODEL_PATH",  "./model/model_mri.keras")
IMAGE_SIZE      = int(os.getenv("IMAGE_SIZE",  "128"))
REDIS_URL       = os.getenv("REDIS_URL",       "redis://localhost:6379/0")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Thiếu OPENAI_API_KEY trong file .env")

# Classes and Meta
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
CLASS_META  = {
    "NonDemented":      ("Không có dấu hiệu sa sút trí tuệ", "#10b981"),
    "VeryMildDemented": ("Sa sút trí tuệ rất nhẹ",            "#f59e0b"),
    "MildDemented":     ("Sa sút trí tuệ mức nhẹ",            "#f97316"),
    "ModerateDemented": ("Sa sút trí tuệ mức trung bình",     "#ef4444"),
}

# Global Clients
oai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
chroma_cli = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_cli.get_or_create_collection(COLLECTION_NAME)

try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
except Exception as e:
    print(f"⚠️ Warning: Could not connect to Redis at {REDIS_URL}. Exception: {e}")
    redis_client = None
