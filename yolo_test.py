import os
import cv2
from ultralytics import YOLO

# Inicjalizacja modelu
model = YOLO('yolov8n.pt')

# AUTOMATYCZNA DETEKCJA ŚRODOWISKA
if os.path.exists('/content'):
    # Logika dla GOOGLE COLAB
    video_path = "/content/drive/MyDrive/Projekt_wizja_komputerowa/data/final_video.mp4"
    output_path = "/content/drive/MyDrive/Projekt_wizja_komputerowa/data/test_frame_result.jpg"
else:
    # Logika dla TWOJEGO PC (Lokalnie w VS Code)
    # Zakładam, że w folderze projektu masz podfolder 'data'
    video_path = "data/final_video.mp4"
    output_path = "data/test_frame_result.jpg"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    print(f"Błąd: Nie znaleziono pliku pod ścieżką: {video_path}")
    exit()

# Detekcja (tylko osoby)
results = model(frame, classes=[0])

# Zapis wyniku
annotated_frame = results[0].plot()
cv2.imwrite(output_path, annotated_frame)
print(f"Sukces! Wynik zapisano w: {output_path}")

cap.release()