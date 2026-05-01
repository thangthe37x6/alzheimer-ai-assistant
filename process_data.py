import fitz
import pdfplumber
import base64
import os
import time
import io
from openai import OpenAI
import json
import chromadb
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


import re

# Tập hợp các token lẻ phổ biến từ OCR sơ đồ / flowchart
_FLOWCHART_TOKENS = {
    'ye', 'có', 'không', 'yes', 'no', 'y', 'n',
    'a', 'b', 'c', 'd', 'e',  # mục lục alphabet lẻ
}

def _filter_short_line(m: re.Match) -> str:
    line = m.group(0).strip()
    if not line:
        return ''
    # Giữ lại nếu có nội dung thật (>4 ký tự sau strip)
    if len(line) > 4:
        return m.group(0)
    # Xóa nếu là token flowchart quen thuộc
    if line.lower() in _FLOWCHART_TOKENS:
        return ''
    # Xóa nếu toàn ký tự đặc biệt (không có chữ/số)
    if re.fullmatch(r'[^\w\d]+', line):
        return ''
    return m.group(0)


def clean_text(text: str) -> str:
    # --- 1. Xóa URL ---
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # --- 2. Xóa header/footer lặp lại (nhiều biến thể dấu/không dấu) ---
    text = re.sub(
        r'H[oộ]i\s+B[eệ]nh\s+Alzheimer.*?Vi[eệ]t\s+Nam[^\n]*\n?',
        '', text, flags=re.IGNORECASE
    )

    # --- 3. Xóa số trang đứng một mình (dòng chỉ có 1-3 chữ số) ---
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    # --- 4a. Xóa citation số dạng 'từ69,70,71' (superscript liền sau chữ) ---
    text = re.sub(
        r'(?<=[a-zA-ZÀ-ỹ\)\.])((?:\d{1,3},){1,6}\d{1,3})(?=[\s,\.;:\n]|$)',
        '', text, flags=re.MULTILINE
    )

    # --- 4b. Xóa footnote đơn cuối từ kiểu 'Alzheimer79' ---
    text = re.sub(
        r'(?<=[a-zA-ZÀ-ỹ])(\d{1,3})(?=[\s,\.;:\n]|$)',
        '', text, flags=re.MULTILINE
    )

    # --- 5. Nối từ bị gạch nối cuối dòng (hyphenation PDF) ---
    # Ví dụ: "Alz-\nheimer" → "Alzheimer"
    text = re.sub(r'-\n(?=[a-zA-ZÀ-ỹ])', '', text)

    # --- 6. Xóa dòng chỉ có bullet/ký tự gạch đặc biệt ---
    text = re.sub(r'^\s*[•\-\*·▪▸►–—]\s*$', '', text, flags=re.MULTILINE)

    # --- 7. Xóa dòng cực ngắn vô nghĩa (flowchart token, OCR artifact) ---
    # Ví dụ: 'ye', 'Có', 'Không', 'C' lẻ loi trên dòng riêng
    text = re.sub(r'^\s*.{1,4}\s*$', _filter_short_line, text, flags=re.MULTILINE)

    # --- 7b. Xóa dòng chỉ gồm các token flowchart lặp lại ---
    # Ví dụ: 'Không Không Có Không' (trang sơ đồ quyết định bị OCR sai)
    _ft_pattern = '|'.join(re.escape(t) for t in _FLOWCHART_TOKENS)
    text = re.sub(
        r'^\s*(?:(?:' + _ft_pattern + r')\s+){1,10}(?:' + _ft_pattern + r')\s*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )

    # --- 8. Xóa ký tự đơn viết hoa lẻ cuối câu (OCR ngắt sai) ---
    # Ví dụ: "...không có thời kỳ bình nguyên kéo dài. C" → xóa " C"
    text = re.sub(r'(?<=[.!?]) [A-ZÀ-Ỹ]\s*(?=\n|$)', ' ', text, flags=re.MULTILINE)

    # ── Nối dòng sau khi đã lọc ────────────────────────────────────────────────

    # --- 9. Nối dòng bị vỡ giữa câu (single newline → space) ---
    # Giữ lại double-newline (đoạn mới thật sự)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # --- 10. Chuẩn hóa nhiều newline liên tiếp → tối đa 2 ---
    text = re.sub(r'\n{3,}', '\n\n', text)

    # --- 11. Chuẩn hóa khoảng trắng ngang thừa (space/tab) ---
    text = re.sub(r'[ \t]+', ' ', text)

    # --- 12. Strip từng dòng (xóa space đầu/cuối mỗi dòng) ---
    text = '\n'.join(line.strip() for line in text.splitlines())

    # --- 13. Chuẩn hóa lại newline (sau các bước trên có thể sinh thêm) ---
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def clean_all_docs(all_docs: list, min_chars: int = 50) -> list:

    cleaned = []
    skipped = 0

    for doc in all_docs:
        cleaned_text = clean_text(doc["text"])
        if len(cleaned_text) >= min_chars:
            cleaned.append({
                "page"  : doc["page"],
                "text"  : cleaned_text,
                "source": doc["source"]
            })
        else:
            skipped += 1

    print(f"Đã clean {len(cleaned)} trang, bỏ {skipped} trang quá ngắn/trống.")
    return cleaned


def chunk_documents(all_docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    for doc in all_docs:
        parts = splitter.split_text(doc["text"])
        for j, part in enumerate(parts):
            chunks.append({
                "id"    : f"page{doc['page']}_chunk{j}",
                "text"  : part,
                "page"  : doc["page"],
                "source": doc["source"]
            })

    print(f"Tổng chunks: {len(chunks)}")
    return chunks

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def build_vector_db(chunks):
    

    chroma = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db"))

    try:
        chroma.delete_collection("alzheimer")
    except:
        pass

    collection = chroma.get_or_create_collection(
        name="alzheimer",
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids        = [c["id"] for c in batch],
            documents  = [c["text"] for c in batch],
            embeddings = [embed_text(c["text"]) for c in batch],
            metadatas  = [{"page": c["page"], "source": c["source"]} for c in batch]
        )

    print("Lưu vào ChromaDB xong!")
    return collection

def load_from_file(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Đã đọc {len(data)} trang từ {input_path}")
    return data



# Đọc từ file json lên
data = load_from_file("parsed_pages.json")

# Clean
data = clean_all_docs(data)

# Chunk
chunks = chunk_documents(data)

# Embed + lưu vào ChromaDB
collection = build_vector_db(chunks)
