"""
Add student CLI.

Behavior:
- Detects face in given image using MediaPipe
- Crops the primary face
- Computes embedding using DeepFace (FaceNet)
- Stores entry in data/students.json

Data types:
- name: str
- image_path: str
- embedding: list[float]
- student: dict
"""
import argparse
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from scripts.utils import load_db, save_db, ensure_db_exists

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5)


def detect_and_crop_first_face(image_path: str) -> np.ndarray:
    """Detect first face and return cropped RGB face image (numpy.ndarray HxWx3)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = mp_face.process(img_rgb)
    if not results.detections:
        raise ValueError("No face detected in the image.")
    # take the first detection
    det = results.detections[0]
    bbox = det.location_data.relative_bounding_box
    h, w, _ = img_rgb.shape
    x = max(int(bbox.xmin * w), 0)
    y = max(int(bbox.ymin * h), 0)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)
    x2 = min(x + bw, w)
    y2 = min(y + bh, h)
    face = img_rgb[y:y2, x:x2]
    # if face is empty, return full image to let DeepFace try
    if face.size == 0:
        return img_rgb
    return face


def compute_embedding_from_rgb(face_rgb: np.ndarray) -> List[float]:
    """
    Compute embedding using DeepFace.
    Input: face_rgb (numpy.ndarray HxWx3, RGB)
    Output: embedding list[float]
    """
    # DeepFace expects BGR by default when passing file path, but accepts numpy RGB if we convert.
    # Convert RGB -> BGR for DeepFace to be safe.
    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    # DeepFace.represent can accept numpy array
    rep = DeepFace.represent(img_path=face_bgr,
                             model_name="Facenet",
                             enforce_detection=False,
                             detector_backend="opencv")
    # DeepFace.represent returns list of dicts when passed an array; handle both cases
    if isinstance(rep, list) and len(rep) > 0:
        emb = rep[0].get("embedding") if isinstance(rep[0], dict) else rep[0]
    elif isinstance(rep, dict) and "embedding" in rep:
        emb = rep["embedding"]
    else:
        # fallback: flatten
        emb = list(np.array(rep).flatten())
    # Ensure embedding is list[float]
    return [float(x) for x in emb]


def add_student(name: str, image_path: str) -> Dict[str, Any]:
    """Add student and return the student dict."""
    ensure_db_exists()
    db = load_db()
    face_rgb = detect_and_crop_first_face(image_path)
    embedding = compute_embedding_from_rgb(face_rgb)  # list[float]
    student_id = str(len(db) + 1)
    student = {"id": student_id, "name": name, "embedding": embedding}
    db.append(student)
    save_db(db)
    return student


def add_student_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Student name")
    parser.add_argument("--image",
                        required=True,
                        help="Path to student image (close-up)")
    args = parser.parse_args()
    try:
        student = add_student(args.name, args.image)
        print(f"✅ Added student: {student['name']} (id={student['id']})")
    except Exception as e:
        print(f"❌ Error adding student: {e}")


if __name__ == "__main__":
    add_student_cli()
