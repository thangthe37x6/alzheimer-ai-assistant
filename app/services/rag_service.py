from typing import List, Dict, Optional
from app.config import oai_client, collection, CLASS_META
from app.services.state import get_chat_history, save_chat_history, get_vit_session

def embed(text: str) -> List[float]:
    return oai_client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

def condense(question: str, history: List[Dict], vit_result: Optional[Dict] = None) -> str:
    vit_context = ""
    if vit_result:
        top   = vit_result["top_class"]
        label = CLASS_META.get(top, (top, ""))[0]
        scores_text = ", ".join(
            f"{c}: {s}%"
            for c, s in sorted(vit_result["scores"].items(), key=lambda x: -x[1])
        )
        vit_context = (
            f"Bệnh nhân trong phiên này vừa được mô hình AI chẩn đoán:\n"
            f"  ▸ Kết quả: {label} ({top})\n"
            f"  ▸ Xác suất: {scores_text}\n\n"
        )
    if not history and not vit_context:
        return question

    hist = "\n".join(
        f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}"
        for m in history[-6:]
    )

    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            f"{vit_context}"
            f"Lịch sử hội thoại:\n{hist}\n\n"
            f"Câu hỏi của user: {question}\n\n"
            f"Nhiệm vụ: Viết lại câu hỏi trên thành một câu truy vấn hoàn chỉnh, "
            f"độc lập, có đủ từ khóa y khoa (tên giai đoạn bệnh, triệu chứng, v.v.) "
            f"để tìm kiếm tài liệu trong cơ sở dữ liệu Alzheimer. "
            f"Chỉ trả về câu truy vấn đã viết lại, không giải thích thêm."}],
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()

def query_rag(question: str, session_id: str, top_k: int = 8):
    history = get_chat_history(session_id)

    # Lấy kết quả ViT của session này (nếu user đã upload ảnh trước đó)
    vit_result = get_vit_session(session_id)

    # Rewrite query — có kèm context ViT nếu có
    condensed  = condense(question, history, vit_result)
    print(f"🔍 [session={session_id}] Condensed RAG query: {condensed}")

    results = collection.query(
        query_embeddings=[embed(condensed)], n_results=top_k
    )

    context = "\n\n".join(results["documents"][0])
    pages   = [m.get("page", "?") for m in results["metadatas"][0]]

    history.append({"role": "user", "content": question})

    resp = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là chuyên gia y khoa về bệnh Alzheimer. "
                    "Chỉ trả lời dựa trên tài liệu được cung cấp. "
                    "Nếu không có thông tin thì nói không biết.\n\n"
                    f"Tài liệu tham khảo:\n{context}"
                ),
            },
            *history,
        ],
        max_tokens=1024,
    )
    answer = resp.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    
    save_chat_history(session_id, history)
    return answer, [str(p) for p in pages]
