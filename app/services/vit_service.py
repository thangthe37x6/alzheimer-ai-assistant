import io
import numpy as np
import keras
from PIL import Image
from typing import Dict
from app.config import VIT_MODEL_PATH, IMAGE_SIZE, CLASS_NAMES
from app.models.vit_layers import Patches, PatchEncoder

def load_model():
    try:
        model = keras.models.load_model(VIT_MODEL_PATH, custom_objects={
            "Patches": Patches,
            "PatchEncoder": PatchEncoder
        })
        print(f"✅ Model loaded: {VIT_MODEL_PATH}")
        print(f"   Input shape : {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"⚠️  Model load failed: {e}")
        return None

vit_model = load_model()

def predict(img_bytes: bytes) -> Dict:
    if vit_model is None:
        raise RuntimeError("Model chưa được load. Kiểm tra VIT_MODEL_PATH trong .env")
        
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    logits = vit_model.predict(arr, verbose=0)[0]
    e     = np.exp(logits - np.max(logits))
    probs = e / e.sum()
    scores    = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))}
    top_class = max(scores, key=scores.get)
    return {"top_class": top_class, "scores": scores}
