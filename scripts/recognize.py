"""
Recognize faces from a group photo.

Behavior:
- Detects faces with MediaPipe
- For each detected face, crops and computes embedding (DeepFace)
- Matches embedding against data/students.json using cosine similarity
- Produces annotated image output at output_recognized.jpg

Data types:
- detected_embeddings: list[list[float]]
- db: list[dict]
"""
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from scripts.utils import load_db, cosine_similarity

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5)

OUTPUT_PATH = Path("output_recognized.jpg")


def detect_faces_with_bboxes(
        img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return list of bounding boxes (x,y,w,h) in pixel coordinates for img_rgb (HxWx3)."""
    results = mp_face.process(img_rgb)
    h, w, _ = img_rgb.shape
    bboxes = []
    if not results.detections:
        return bboxes
    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x = max(int(bbox.xmin * w), 0)
        y = max(int(bbox.ymin * h), 0)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        x2 = min(x + bw, w)
        y2 = min(y + bh, h)
        # clip
        bboxes.append((x, y, x2 - x, y2 - y))
    return bboxes


def compute_embedding_from_rgb(face_rgb: np.ndarray) -> List[float]:
    """Compute embedding using DeepFace given an RGB crop."""
    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    rep = DeepFace.represent(img_path=face_bgr,
                             model_name="Facenet",
                             enforce_detection=False,
                             detector_backend="opencv")
    if isinstance(rep, list) and len(rep) > 0:
        emb = rep[0].get("embedding") if isinstance(rep[0], dict) else rep[0]
    elif isinstance(rep, dict) and "embedding" in rep:
        emb = rep["embedding"]
    else:
        emb = list(np.array(rep).flatten())
    return [float(x) for x in emb]


def recognize_from_group(image_path: str, threshold: float = 0.5) -> None:
    db = load_db()
    if len(db) == 0:
        print("No enrolled students found. Add students first.")
        return

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Image not found: {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    bboxes = detect_faces_with_bboxes(img_rgb)
    print(f"Detected {len(bboxes)} faces.")

    annotated = img_bgr.copy()

    for i, (x, y, w_box, h_box) in enumerate(bboxes, start=1):
        # Expand bounding box slightly for better crops
        pad = int(0.15 * max(w_box, h_box))
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w_box + pad, img_rgb.shape[1])
        y1 = min(y + h_box + pad, img_rgb.shape[0])
        crop = img_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        try:
            emb = compute_embedding_from_rgb(crop)
        except Exception as e:
            print(f"Warning: failed to compute embedding for face {i}: {e}")
            continue

        best_score = -1.0
        best_name = "Unknown"
        for s in db:
            score = cosine_similarity(emb, s["embedding"])
            if score > best_score:
                best_score = score
                best_name = s["name"]

        label = best_name if best_score >= threshold else "Unknown"
        print(f"Face {i} -> {label} (score={best_score:.2f})")

        # annotate on image (BGR)
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label} {best_score:.2f}",
                    (x0, max(y0 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    cv2.imwrite(str(OUTPUT_PATH), annotated)
    print(f"✅ Annotated image saved to {OUTPUT_PATH}")


def recognize_from_group_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to group photo")
    parser.add_argument("--threshold",
                        type=float,
                        default=0.5,
                        help="Cosine similarity threshold")
    args = parser.parse_args()
    recognize_from_group(args.image, args.threshold)


if __name__ == "__main__":
    recognize_from_group_cli()
