import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# TUTAJ WKLEJ DOKŁADNIE TĘ ŚCIEŻKĘ, KTÓRA ZADZIAŁAŁA W DIAGNOSTYCE:
video_path = "/content/drive/MyDrive/Projekt_wizja_komputerowa/data/final_video.mp4"

# TUTAJ WKLEJ ŚCIEŻKĘ WYJŚCIOWĄ DO ZAPISU (TEN SAM FOLDER, INNY PLIK):
output_path = "/content/drive/MyDrive/Projekt_wizja_komputerowa/test_frame_result.jpg"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    print("Krytyczny błąd: Nie można odczytać klatki z pliku wideo.")
    exit()

# Uruchomienie detekcji (tylko klasa 'osoba')
results = model(frame, classes=[0])
boxes = results[0].boxes
print(f"Sukces operacyjny: Wykryto {len(boxes)} zawodników na wyciągniętej klatce.")

# Rysowanie ramek i zapis na stały Dysk Google
annotated_frame = results[0].plot()
cv2.imwrite(output_path, annotated_frame)
print(f"Plik weryfikacyjny został zapisany bezpośrednio na Dysku Google w: {output_path}")

cap.release()
