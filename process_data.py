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
_FLOWCHART_TOKENS = {
    'ye', 'có', 'không', 'yes', 'no', 'y', 'n',
    'a', 'b', 'c', 'd', 'e',  # mục lục alphabet lẻ
}

def _filter_short_line(m: re.Match) -> str:
    line = m.group(0).strip()
    if not line:
        return ''
    if len(line) > 4:
        return m.group(0)
    if line.lower() in _FLOWCHART_TOKENS:
        return ''
    if re.fullmatch(r'[^\w\d]+', line):
        return ''
    return m.group(0)


def clean_text(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(
        r'H[oộ]i\s+B[eệ]nh\s+Alzheimer.*?Vi[eệ]t\s+Nam[^\n]*\n?',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(
        r'(?<=[a-zA-ZÀ-ỹ\)\.])((?:\d{1,3},){1,6}\d{1,3})(?=[\s,\.;:\n]|$)',
        '', text, flags=re.MULTILINE
    )
    text = re.sub(
        r'(?<=[a-zA-ZÀ-ỹ])(\d{1,3})(?=[\s,\.;:\n]|$)',
        '', text, flags=re.MULTILINE
    )
    text = re.sub(r'-\n(?=[a-zA-ZÀ-ỹ])', '', text)
    text = re.sub(r'^\s*[•\-\*·▪▸►–—]\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*.{1,4}\s*$', _filter_short_line, text, flags=re.MULTILINE)
    _ft_pattern = '|'.join(re.escape(t) for t in _FLOWCHART_TOKENS)
    text = re.sub(
        r'^\s*(?:(?:' + _ft_pattern + r')\s+){1,10}(?:' + _ft_pattern + r')\s*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )
    text = re.sub(r'(?<=[.!?]) [A-ZÀ-Ỹ]\s*(?=\n|$)', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = '\n'.join(line.strip() for line in text.splitlines())
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
data = load_from_file("parsed_pages.json")
data = clean_all_docs(data)
chunks = chunk_documents(data)
collection = build_vector_db(chunks)
