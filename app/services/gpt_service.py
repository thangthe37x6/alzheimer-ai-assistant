import base64
from typing import Dict
import io
from PIL import Image
from app.config import oai_client, CLASS_META

def analyze_with_gpt4v(img_bytes: bytes, vit_result: Dict) -> str:
    # Chuyển ảnh về chuẩn JPEG để tránh lỗi image_parse_error của OpenAI nếu up PNG
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((1024, 1024)) # Resize để tiết kiệm token
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    top   = vit_result["top_class"]
    label = CLASS_META.get(top, (top, ""))[0]
    scores_text = "\n".join(
        f"  • {c}: {s}%"
        for c, s in sorted(vit_result["scores"].items(), key=lambda x: -x[1])
    )
    prompt = f"""[THÔNG TIN NGHIÊN CỨU/HỌC THUẬT]
        Đây là một ca mô phỏng hình ảnh học (không phải chẩn đoán bệnh nhân thực tế). Hãy đóng vai một trợ lý nghiên cứu phân tích ảnh MRI não.

        Mô hình học máy (ViT) đã đưa ra kết quả dự đoán ban đầu cho bức ảnh này:
        ▸ Nhãn dự đoán: {label} ({top})
        ▸ Xác suất:
        {scores_text}

        Hãy cung cấp một báo cáo quan sát khách quan mang tính học thuật theo cấu trúc sau:

        chỉ cần nêu tóm tắt giai đoạn này '{top}' của bệnh thôi. 

        Bắt buộc kết thúc bằng dòng: "⚠️ Lưu ý: Đây là phân tích mô phỏng từ AI, không có giá trị thay thế chẩn đoán y khoa từ bác sĩ chuyên môn."."""

    resp = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=400,
    )
    return resp.choices[0].message.content
