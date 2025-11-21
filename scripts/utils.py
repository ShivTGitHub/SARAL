"""
Utility functions used by add_student.py and recognize.py

Data types:
- Image: numpy.ndarray
- Embedding: list[float]
- DB entry: dict
"""
from pathlib import Path
import json
from typing import List, Dict, Any
import numpy as np

DATA_PATH = Path("data/students.json")


def ensure_db_exists() -> None:
    """Ensure the students.json file exists. (Type: None -> None)"""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        DATA_PATH.write_text("[]", encoding="utf-8")


def load_db() -> List[Dict[str, Any]]:
    """Return the student DB as a list of dicts. (Type: -> list[dict])"""
    ensure_db_exists()
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_db(db: List[Dict[str, Any]]) -> None:
    """Save DB to disk. (Type: list[dict] -> None)"""
    with DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors. (Type: list[float],list[float] -> float)"""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    if np.linalg.norm(a_arr) == 0 or np.linalg.norm(b_arr) == 0:
        return 0.0
    return float(
        np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
