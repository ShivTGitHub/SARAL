"""
Main entry for the Replit prototype.

Data types referenced:
- students_db: list[dict] where dict ~ {'id': str, 'name': str, 'embedding': list[float]}
- embedding: list[float]
- img: numpy.ndarray (HxWx3)
"""
from scripts.add_student import add_student_cli
from scripts.recognize import recognize_from_group_cli


def main() -> None:
    print("Facial recognition attendance prototype (Replit).")
    print("Usage examples:")
    print(
        "  python -m scripts.add_student --name Alice --image students/alice.jpg"
    )
    print("  python -m scripts.recognize --image group_photos/class1.jpg")


if __name__ == "__main__":
    main()
