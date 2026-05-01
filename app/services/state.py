import json
from typing import Dict, List, Optional
from app.config import redis_client

SESSION_TTL = 3600

_fallback_sessions: Dict[str, List[Dict]] = {}
_fallback_vit: Dict[str, Dict] = {}

def save_vit_session(session_id: str, vit_result: Dict) -> None:
    if redis_client:
        redis_client.setex(f"vit:{session_id}", SESSION_TTL, json.dumps(vit_result))
    else:
        _fallback_vit[session_id] = vit_result

def get_vit_session(session_id: str) -> Optional[Dict]:
    if redis_client:
        data = redis_client.get(f"vit:{session_id}")
        if data:
            return json.loads(data)
        return None
    else:
        return _fallback_vit.get(session_id)

def save_chat_history(session_id: str, history: List[Dict]) -> None:
    if redis_client:
        redis_client.setex(f"chat:{session_id}", SESSION_TTL, json.dumps(history))
    else:
        _fallback_sessions[session_id] = history

def get_chat_history(session_id: str) -> List[Dict]:
    if redis_client:
        data = redis_client.get(f"chat:{session_id}")
        if data:
            return json.loads(data)
        return []
    else:
        return _fallback_sessions.get(session_id, [])

def clear_session(session_id: str) -> None:
    if redis_client:
        redis_client.delete(f"vit:{session_id}", f"chat:{session_id}")
    else:
        _fallback_sessions.pop(session_id, None)
        _fallback_vit.pop(session_id, None)

def get_active_sessions_count() -> int:
    if redis_client:
        return len(redis_client.keys("chat:*"))
    else:
        return len(_fallback_sessions)

