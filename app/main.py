from fastapi import FastAPI
from app.routes import home, analyze, chat, health

app = FastAPI(title="Alzheimer AI — ViT + RAG")

app.include_router(home.router)
app.include_router(analyze.router)
app.include_router(chat.router)
app.include_router(health.router)
