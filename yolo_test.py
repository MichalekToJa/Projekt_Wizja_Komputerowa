import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def get_dominant_color(crop_bgr):
    # Ekstrakcja matematycznej dominanty
    if crop_bgr.size == 0:
        return np.array([0, 0, 0], dtype=np.float32)

    hsv_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    pixels = np.float32(hsv_crop.reshape(-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

    _, counts = np.unique(labels, return_counts=True)
    return centers[np.argmax(counts)]

# --- INICJALIZACJA ŚRODOWISKA ---
model = YOLO('yolov8m.pt')

if os.path.exists('/content'):
    video_path = "/content/drive/MyDrive/Projekt_wizja_komputerowa/data/final_video.mp4"
    output_path = "/content/drive/MyDrive/Projekt_wizja_komputerowa/data/output_video_latched.mp4"
else:
    video_path = "data/final_video.mp4"
    output_path = "data/output_video_latched.mp4"

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Struktury pamięci dla obiektów
player_color_history = defaultdict(list)
locked_colors = {}
LATCH_FRAME_COUNT = 45 # Moment zamrożenia wektora barwy (ok. 1.5 sekundy)

print("Rozpoczynam przetwarzanie strumienia z twardym zatrzaskiem tożsamości...")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    # Detekcja z podtrzymaniem ID
    results = model.track(frame, classes=[0], persist=True, conf=0.35, iou=0.65, verbose=False)

    player_crops_data = []
    active_hsv_list = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        # ETAP 1: Analiza wektorów z uwzględnieniem zatrzasku
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Pobieranie próbek z obrazu TYLKO dla niezablokowanych ID
            if track_id not in locked_colors:
                w = x2 - x1
                h = y2 - y1
                y_start = max(0, int(y1 + h * 0.3))
                y_end = min(frame.shape[0], int(y2 - h * 0.3))
                x_start = max(0, int(x1 + w * 0.3))
                x_end = min(frame.shape[1], int(x2 - w * 0.3))
                
                crop = frame[y_start:y_end, x_start:x_end]
                dom_color = get_dominant_color(crop)
                
                player_color_history[track_id].append(dom_color)
                
                # Inicjacja zatrzasku
                if len(player_color_history[track_id]) == LATCH_FRAME_COUNT:
                    locked_colors[track_id] = np.median(player_color_history[track_id], axis=0)
                    # Optymalizacja pamięci RAM
                    del player_color_history[track_id]

            # Ustalenie aktywnego wektora do kategoryzacji drużynowej
            if track_id in locked_colors:
                current_vector = locked_colors[track_id]
            else:
                current_vector = np.median(player_color_history[track_id], axis=0)

            player_crops_data.append((x1, y1, x2, y2, track_id))
            active_hsv_list.append(current_vector)

    # ETAP 2: Grupowanie K-Means
    if len(active_hsv_list) > 2:
        hsv_array = np.array(active_hsv_list, dtype=np.float32)
        
        n_clusters = min(3, len(active_hsv_list))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, team_labels, team_centers = cv2.kmeans(hsv_array, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        labels_flat = team_labels.flatten()
        cluster_counts = np.bincount(labels_flat, minlength=n_clusters)
        sorted_clusters = np.argsort(cluster_counts)[::-1]
        
        main_team_1_id = sorted_clusters[0]
        main_team_2_id = sorted_clusters[1]
        
        for i, (x1, y1, x2, y2, track_id) in enumerate(player_crops_data):
            cluster_id = int(labels_flat[i])
            
            # Wskaźnik interfejsu (UX) podczas debugowania wideo
            status_flag = "[LOCKED]" if track_id in locked_colors else "[...]"
            
            if cluster_id == main_team_1_id:
                label_text = f"Druzyna 1 (ID:{track_id}) {status_flag}"
            elif cluster_id == main_team_2_id:
                label_text = f"Druzyna 2 (ID:{track_id}) {status_flag}"
            else:
                label_text = f"Unknown (ID:{track_id}) {status_flag}"
                
            team_hsv_center = np.uint8([[team_centers[cluster_id]]])
            team_bgr_color = cv2.cvtColor(team_hsv_center, cv2.COLOR_HSV2BGR)[0][0]
            color_tuple = (int(team_bgr_color[0]), int(team_bgr_color[1]), int(team_bgr_color[2]))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_tuple, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tuple, 2)

    out.write(frame)
    
    if frame_count % 30 == 0:
         print(f"Przetworzono {frame_count} klatek...")

cap.release()
out.release()
print(f"Sukces operacyjny. Wideo z zatrzaskiem tożsamości zapisane w: {output_path}")