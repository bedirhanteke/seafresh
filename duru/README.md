# Kamera Watchdog (Senaryo 4)

RTSP kameraların **bağlantı kopması**, **donmuş görüntü (freeze)** ve **kurcalanma
(tamper)** durumlarını takip eden, YOLO/görüntü işleme pipeline'ından bağımsız,
hafif bir arka plan servisi. Her kamera için ayrı bir thread açar; durumu
`camera_health_status.json` (anlık) ve `camera_health_log.csv` (olay geçmişi)
dosyalarına yazar.

Bu script, NVR'ın kendi heartbeat/tamper/video-loss özellikleri yetersiz
kaldığında kullanılacak bir yedek katman olarak tasarlandı. Öncelik her zaman
NVR'ın kendi Setup → Event → Video Detection ve Network → Alarm ayarlarıdır.

## Kurulum

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Ayarlar

Gerçek NVR bilgileri (IP, kullanıcı adı, şifre) **repoya commit edilmez**
(bkz. `.gitignore`). Kendi ortamında çalıştırmak için:

```bash
cp watchdog_config.example.json watchdog_config.json
```

sonra `watchdog_config.json` içindeki `nvr_host`, `username`, `password` ve
`cameras` listesini kendi NVR'ına göre doldur.

Şifreyi dosyaya yazmak yerine ortam değişkeniyle de verebilirsin (dosyadaki
değerin önüne geçer):

```bash
export NVR_PASSWORD="..."
```

### Önemli config alanları

- `cameras`: izlenecek her kamera için `name` (kendi seçtiğin etiket) ve
  `channel` (NVR'daki RTSP kanal numarası).
- `freeze.diff_threshold` / `freeze_seconds`: ardışık kareler arası fark bu
  eşiğin altında bu süre boyunca kalırsa görüntü "donmuş" sayılır. Eşiği kendi
  kameranda ölçüp ayarlaman gerekir (statik ama canlı bir sahnede bile
  sensör/sıkıştırma gürültüsü yüzünden fark genelde 0'a tam inmez).
- `tamper.spatial_std_threshold` / `baseline_diff_threshold` /
  `tamper_seconds`: NVR'da native tamper alarmı yoksa kullanılan
  görüntü-tabanlı iki sinyal (lens kapatma/boyama ve kamera yönü değişikliği).
- `poll.*`: okuma aralığı, yeniden bağlanma gecikmesi, açma/okuma zaman aşımı,
  RTSP transport (`tcp` önerilir).

## Çalıştırma

```bash
python main.py
# farklı bir config dosyasıyla:
python main.py --config baska_config.json
```

Ctrl+C ile durdurulabilir (tüm thread'ler düzgün kapatılır, son durum dosyaya
yazılır).

## Çıktılar

- `camera_health_status.json`: her kamera için anlık durum
  (`OK` / `DISCONNECTED` / `FROZEN` / `TAMPER` / `STARTING`), son kare yaşı,
  bağlantı zamanı, son diff/tamper skorları.
- `camera_health_log.csv`: durum değişikliği geçmişi (`timestamp, camera,
  channel, status, diff_score`).

Her ikisi de `.gitignore` ile hariç tutulmuştur, çalışma zamanında üretilir.
