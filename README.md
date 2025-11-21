# Facial Recognition Attendance - Replit Prototype (Template)

This template creates a simple project scaffold to prototype face enrollment and recognition on Replit.

## Folders
- `students/` : place enrollment images here (one folder image per student is fine to start)
- `group_photos/` : place classroom photos here for recognition tests
- `models/` : tflite/onnx models
- `data/` : stores `students.json` with embeddings
- `scripts/` : helper scripts to add students and run recognition
- `logs/` : attendance logs (e.g., daily json files)

## Quickstart
1. Upload student images to `students/` (e.g., `students/alice.jpg`)
2. Run: `python -m scripts.add_student --name Alice --image students/alice.jpg`
3. Add more students
4. Run: `python -m scripts.recognize --image group_photos/class1.jpg`

## Data types used (examples)
- `students.json` : list[dict] where each dict has keys: `id` (str), `name` (str), `embedding` (list[float])
- Embeddings are represented as `list[float]` (e.g., length 128)


# Local Offline Facial Recognition Prototype

This project is a Python prototype for an offline facial-recognition attendance system intended to run on mobile devices.

## Features
- Student enrollment via image embedding
- Face detection using MediaPipe
- Embedding generation using DeepFace
- Attendance recognition from group photos
- Local JSON database

## Folders
- /students
- /group_photos
- /scripts
- /data
- /output

## How to Run
python -m scripts.add_student --name "..." --image "students/xxx.jpg"
python -m scripts.recognize --image "group_photos/class1.jpg"

