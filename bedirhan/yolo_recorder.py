# pyrefly: ignore [missing-import]
import cv2
import time
import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO('yolov8s.pt')

# 1. Müşteri (Customer) ROI (Önceki kırpılmış hali)
roi_customer_norm = [
    (1038.0097 / 3072, 450.0000 / 2048),
    (1119.9297 / 3072, 733.2839 / 2048),
    (1731.3326 / 3072, 647.3678 / 2048),
    (1679.3834 / 3072, 499.5121 / 2048),
    (1493.5648 / 3072, 459.5512 / 2048),
    (1377.6780 / 3072, 420.0000 / 2048),
    (1185.8653 / 3072, 410.0000 / 2048)
]

# 2. Kasiyer (Cashier) ROI (Kullanıcı isteğiyle 100px yukarı, 50px sağa/sola büyütüldü)
roi_cashier_norm = [
    (1083.8673 / 3072, 1663.2224 / 2048),  # 100px yukarı, 50px sola
    (2242.9307 / 3072, 1349.5288 / 2048),  # 100px yukarı, 50px sağa
    (2392.7844 / 3072, 2048.0000 / 2048),  # 50px sağa (Zaten alt sınırda)
    (1075.8751 / 3072, 2048.0000 / 2048)   # 50px sola (Zaten alt sınırda)
]

import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# RTSP Kamera / Video Bağlantısı
DEFAULT_RTSP = "rtsp://your5eyt:rfg34hg-6he@77.44.64.69:554/cam/playback?channel=6&subtype=0&starttime=2026_07_21_15_00_00&endtime=2026_07_21_18_00_00"

if len(sys.argv) > 1:
    RTSP_URL = sys.argv[1]
else:
    RTSP_URL = os.getenv("RTSP_URL", "").strip()
    if not RTSP_URL or RTSP_URL == " ":
        RTSP_URL = DEFAULT_RTSP

print(f"RTSP Kamerasına bağlanılıyor... Lütfen bekleyin: {RTSP_URL}")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print(f"Hata: Video kaynağına/RTSP akışına bağlanılamadı! (URL: {RTSP_URL})")
    sys.exit(1)

ret, first_frame = cap.read()
if not ret or first_frame is None:
    print(f"Hata: Kameradan ilk kare okunamadı! (URL: {RTSP_URL})")
    cap.release()
    sys.exit(1)

frame_height, frame_width = first_frame.shape[:2]
fps = int(cap.get(cv2.CAP_PROP_FPS))

if fps <= 0:
    fps = 25
    
delay = int(1000 / fps)

save_dir = "yakalanan_insanlar"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# --- KAYIT VE MANTIK DEĞİŞKENLERİ ---
recording = False
record_timeout = 7.0  # Kişi çıktıktan sonra çekimi durdurma payı
last_seen_time = 0

wait_before_record = 7.0  # Müşteri sipariş verip ödeme aşamasına geçene kadar beklenen süre (7 saniye)
person_first_seen_time = 0

# Fotoğraf çekme ayarları (Saniyede 4 fotoğraf)
last_capture_time = 0
capture_interval = 1.0 / 4.0  
image_counter = 0
session_id = ""

print("Sistem başlatıldı.")
print("Kayıt için HEM MÜŞTERİ HEM DE KASİYER'in kendi alanında olması gerekiyor.")

window_name = "YOLOv8 Akilli Kamera Sistemi"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

# Müşteri Maskesi
roi_customer_pts = np.array([[int(x * frame_width), int(y * frame_height)] for x, y in roi_customer_norm], np.int32).reshape((-1, 1, 2))
mask_customer = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv2.fillPoly(mask_customer, [roi_customer_pts], 255)
area_customer = cv2.countNonZero(mask_customer)
if area_customer == 0: area_customer = 1

# Kasiyer Maskesi
roi_cashier_pts = np.array([[int(x * frame_width), int(y * frame_height)] for x, y in roi_cashier_norm], np.int32).reshape((-1, 1, 2))
mask_cashier = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv2.fillPoly(mask_cashier, [roi_cashier_pts], 255)
area_cashier = cv2.countNonZero(mask_cashier)
if area_cashier == 0: area_cashier = 1

has_first_frame = True

while True:
    if has_first_frame:
        frame = first_frame
        ret = True
        has_first_frame = False
    else:
        ret, frame = cap.read()

    if not ret or frame is None:
        print("Kameradan görüntü alınamadı veya yayın sona erdi!")
        break
        
    clean_frame = frame.copy()
    
    # Ekrana Müşteri ve Kasiyer bölgelerini çiz
    cv2.polylines(frame, [roi_customer_pts], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(frame, "Musteri Alani", tuple(roi_customer_pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.polylines(frame, [roi_cashier_pts], isClosed=True, color=(255, 0, 255), thickness=2)
    cv2.putText(frame, "Kasiyer Alani", tuple(roi_cashier_pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    results = model(frame, verbose=False)
    
    customer_detected = False
    cashier_detected = False
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls[0]) == 0:  # Sadece insan
                x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                
                # Sınır kontrolü
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                
                person_area = (x2 - x1) * (y2 - y1)
                
                if person_area > 0:
                    # Müşteri maskesiyle kesişim
                    overlap_customer = cv2.countNonZero(mask_customer[y1:y2, x1:x2]) / area_customer
                    # Kasiyer maskesiyle kesişim
                    overlap_cashier = cv2.countNonZero(mask_cashier[y1:y2, x1:x2]) / area_cashier
                    
                    if overlap_customer >= 0.40:
                        customer_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"MUSTERI (%{int(overlap_customer*100)})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    elif overlap_cashier >= 0.40:
                        cashier_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(frame, f"KASIYER (%{int(overlap_cashier*100)})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        # Ekranı çok doldurmamak için sadece küçük bir kesişim varsa yazdıralım
                        if overlap_customer > 0 or overlap_cashier > 0:
                            cv2.putText(frame, "Geciyor", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    current_time = time.time()
    
    # Hem Müşteri Hem Kasiyer alanında insan varsa (TRUE AND TRUE)
    if customer_detected and cashier_detected:
        last_seen_time = current_time
        if person_first_seen_time == 0:
            person_first_seen_time = current_time
            
        if not recording and (current_time - person_first_seen_time >= wait_before_record):
            print("Hem Müşteri hem Kasiyer hazır, fotoğraf çekimi başlıyor...")
            recording = True
            session_id = time.strftime("%Y%m%d-%H%M%S")
            image_counter = 0
    else:
        # Müşteri 7 saniye beklerken cüzdan almak için eğilirse ve yapay zeka onu anlık kaybederse, 
        # sayacın baştan başlamaması (sıfırlanmaması) için kayıp toleransını 1.0'dan 2.5 saniyeye çıkardık.
        if not recording and (current_time - last_seen_time > 2.5):
            person_first_seen_time = 0
            
    if recording:
        if current_time - last_capture_time >= capture_interval:
            filename = os.path.join(save_dir, f"islem_{session_id}_{image_counter:04d}.jpg")
            cv2.imwrite(filename, clean_frame)
            last_capture_time = current_time
            image_counter += 1
        
        if current_time - last_seen_time > record_timeout:
            print(f"İşlem bitti. Toplam {image_counter} fotoğraf kaydedildi.")
            recording = False
            person_first_seen_time = 0
            
    if recording:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, f"FOTOGRAF CEKILIYOR ({image_counter})", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif person_first_seen_time > 0 and not recording:
        beklenen_sure = current_time - person_first_seen_time
        kalan = max(0, wait_before_record - beklenen_sure)
        cv2.putText(frame, f"Bekleniyor: {kalan:.1f}s", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
    cv2.imshow(window_name, frame)
    
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        manuel_session_id = time.strftime("%Y%m%d-%H%M%S")
        manuel_filename = os.path.join(save_dir, f"manuel_{manuel_session_id}.jpg")
        cv2.imwrite(manuel_filename, clean_frame)
        print(f"Manuel fotoğraf çekildi: {manuel_filename}")

cap.release()
cv2.destroyAllWindows()
