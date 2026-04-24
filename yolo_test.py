import cv2
from ultralytics import YOLO

# 1. Inicjalizacja modelu
# Używamy wersji 'yolov8n.pt' (Nano). Jest to najlżejszy wariant.
# Przyczyna: Wersja Nano ładuje się do pamięci VRAM w zaledwie kilkaset megabajtów.
# Skutek: Gwarantuje to brak awarii na karcie GTX 1660 Super (6GB), zachowując wysoką prędkość.
model = YOLO('yolov8n.pt')

# 2. Wczytanie przygotowanego wideo
video_path = "data/test_video.mp4"
cap = cv2.VideoCapture(video_path)

# 3. Ekstrakcja pojedynczej klatki do testów
ret, frame = cap.read()
if not ret:
    print("Krytyczny błąd: Nie można odczytać klatki z pliku wideo.")
    exit()

# 4. Uruchomienie detekcji (Inference)
# Przekazujemy klatkę do modelu. classes=[0] wymusza detekcję wyłącznie klasy 'person' (osoba),
# ignorując piłkę, trybuny czy reklamy, co drastycznie oszczędza moc obliczeniową.
results = model(frame, classes=[0])

# 5. Wyodrębnienie współrzędnych (Bounding Boxes)
# YOLO zwraca wyniki w formie złożonego obiektu. Musimy wyciągnąć z niego surowe macierze.
boxes = results[0].boxes
print(f"Wykryto {len(boxes)} zawodników/osób na pierwszej klatce.")

for box in boxes:
    # Pobranie koordynatów [x_min, y_min, x_max, y_max] w formacie liczb całkowitych
    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
    print(f"Koordynaty zawodnika: [{x1}, {y1}, {x2}, {y2}]")

# 6. Zapisanie klatki z wyrysowanymi obrysami do celów weryfikacji wizualnej
annotated_frame = results[0].plot()
cv2.imwrite("data/test_frame_result.jpg", annotated_frame)
print("Plik weryfikacyjny został zapisany jako 'data/test_frame_result.jpg'.")

cap.release()