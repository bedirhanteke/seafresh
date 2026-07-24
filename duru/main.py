#!/usr/bin/env python3
"""Kamera sağlık takibi (Senaryo 4): RTSP bağlantı + donmuş kare (freeze) watchdog.

YOLO/görüntü işleme pipeline'ından bağımsız, ayrı ve hafif bir process olarak
çalışır. Her kamera için ayrı thread açar; bağlantı kopmasını ve donmuş kareyi
tespit edip camera_health_status.json (anlık durum) ve camera_health_log.csv
(uptime/olay geçmişi) dosyalarına yazar.

Not: Bu script, NVR'ın native heartbeat/tamper/video-loss özellikleri yetersiz
kaldığında kullanılacak yedek katman olarak tasarlandı (bkz. proje notları).
Öncelik her zaman NVR'ın Setup->Event->Video Detection ve Network->Alarm
ayarlarıdır.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    print(
        "Eksik paket var. Once su komutu calistir:\n"
        "  python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


DEFAULT_CONFIG_PATH = Path(__file__).with_name("watchdog_config.json")

STATUS_OK = "OK"
STATUS_DISCONNECTED = "DISCONNECTED"
STATUS_FROZEN = "FROZEN"
STATUS_TAMPER = "TAMPER"
STATUS_STARTING = "STARTING"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_rtsp_url(host: str, port: int, username: str, password: str, channel: int, subtype: int) -> str:
    return (
        f"rtsp://{username}:{password}@{host}:{port}"
        f"/cam/realmonitor?channel={channel}&subtype={subtype}"
    )


@dataclass
class CameraConfig:
    name: str
    channel: int


@dataclass
class WatchdogSettings:
    nvr_host: str
    nvr_port: int
    username: str
    password: str
    subtype: int
    cameras: list[CameraConfig]
    diff_threshold: float
    freeze_seconds: float
    spatial_std_threshold: float
    baseline_diff_threshold: float
    tamper_seconds: float
    read_interval_seconds: float
    log_interval_seconds: float
    reconnect_delay_seconds: float
    open_timeout_ms: int
    read_timeout_ms: int
    transport: str
    status_json_path: Path
    log_csv_path: Path


def load_settings(config_path: Path) -> WatchdogSettings:
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)

    password = os.environ.get("NVR_PASSWORD", raw.get("password", ""))
    cameras = [
        CameraConfig(name=cam["name"], channel=int(cam["channel"]))
        for cam in raw.get("cameras", [])
    ]
    if not cameras:
        raise ValueError("config icinde 'cameras' listesi bos olamaz.")

    freeze = raw.get("freeze", {})
    tamper = raw.get("tamper", {})
    poll = raw.get("poll", {})
    output = raw.get("output", {})

    base_dir = config_path.parent
    return WatchdogSettings(
        nvr_host=raw["nvr_host"],
        nvr_port=int(raw.get("nvr_port", 554)),
        username=raw.get("username", ""),
        password=password,
        subtype=int(raw.get("subtype", 1)),
        cameras=cameras,
        diff_threshold=float(freeze.get("diff_threshold", 0.3)),
        freeze_seconds=float(freeze.get("freeze_seconds", 15.0)),
        spatial_std_threshold=float(tamper.get("spatial_std_threshold", 12.0)),
        baseline_diff_threshold=float(tamper.get("baseline_diff_threshold", 8.0)),
        tamper_seconds=float(tamper.get("tamper_seconds", 8.0)),
        read_interval_seconds=float(poll.get("read_interval_seconds", 1.0)),
        log_interval_seconds=float(poll.get("log_interval_seconds", 30.0)),
        reconnect_delay_seconds=float(poll.get("reconnect_delay_seconds", 3.0)),
        open_timeout_ms=int(poll.get("open_timeout_ms", 8000)),
        read_timeout_ms=int(poll.get("read_timeout_ms", 8000)),
        transport=str(poll.get("transport", "tcp")),
        status_json_path=base_dir / output.get("status_json", "camera_health_status.json"),
        log_csv_path=base_dir / output.get("log_csv", "camera_health_log.csv"),
    )


@dataclass
class CameraState:
    name: str
    channel: int
    status: str = STATUS_STARTING
    last_frame_time: Optional[float] = None
    connected_since: Optional[float] = None
    last_diff_score: Optional[float] = None
    last_spatial_std: Optional[float] = None
    last_baseline_diff: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            last_frame_age = (
                round(time.time() - self.last_frame_time, 1)
                if self.last_frame_time is not None
                else None
            )
            return {
                "channel": self.channel,
                "status": self.status,
                "last_frame_age_seconds": last_frame_age,
                "connected_since": (
                    datetime.fromtimestamp(self.connected_since, tz=timezone.utc).isoformat(timespec="seconds")
                    if self.connected_since
                    else None
                ),
                "last_diff_score": self.last_diff_score,
                "last_spatial_std": self.last_spatial_std,
                "last_baseline_diff": self.last_baseline_diff,
                "updated_at": now_iso(),
            }


class CameraWatcher(threading.Thread):
    def __init__(self, cam: CameraConfig, settings: WatchdogSettings, state: CameraState, log_writer, stop_event: threading.Event):
        super().__init__(name=f"watcher-{cam.name}", daemon=True)
        self.cam = cam
        self.settings = settings
        self.state = state
        self.log_writer = log_writer
        self._stop_event = stop_event
        self._prev_gray: Optional[np.ndarray] = None
        self._baseline_gray: Optional[np.ndarray] = None
        self._low_diff_since: Optional[float] = None
        self._low_std_since: Optional[float] = None
        self._high_baseline_diff_since: Optional[float] = None
        self._last_logged_status: Optional[str] = None
        self._last_log_time = 0.0

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.settings.transport}"
        url = build_rtsp_url(
            self.settings.nvr_host,
            self.settings.nvr_port,
            self.settings.username,
            self.settings.password,
            self.cam.channel,
            self.settings.subtype,
        )
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.settings.open_timeout_ms)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.settings.read_timeout_ms)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            return None
        return capture

    def _set_status(self, status: str) -> None:
        with self.state.lock:
            changed = self.state.status != status
            self.state.status = status
            if status == STATUS_OK and self.state.connected_since is None:
                self.state.connected_since = time.time()
            if status == STATUS_DISCONNECTED:
                self.state.connected_since = None
        if changed:
            self._write_log(status)

    def _write_log(self, status: str) -> None:
        self.log_writer(self.cam.name, self.cam.channel, status, self.state.last_diff_score)
        self._last_logged_status = status
        self._last_log_time = time.time()

    def _analyze_frame(self, frame: np.ndarray) -> None:
        """Her karede donma (freeze) ve kurcalanma (tamper) sinyallerini degerlendirir.

        Tamper icin NVR/kamera native destegi yok (sadece RTSP erisimimiz var),
        bu yuzden iki goruntu-tabanli sinyale bakiyoruz:
          - spatial_std: karenin kendi ic varyansi cok dusukse (uniform siyah/gri)
            lens kapatilmis/boyanmis olabilir.
          - baseline_diff: yavas guncellenen bir referans kareden surdurulebilir
            sekilde uzaklasma, kameranin yonunun degistirildigini gosterebilir.
        Ikisi de kisa sureli hareketle (insan gecisi) karistirilmamak icin
        tamper_seconds boyunca surmesi sartiyla tetiklenir.
        """
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        now = time.time()

        spatial_std = float(np.std(gray))
        with self.state.lock:
            self.state.last_spatial_std = round(spatial_std, 2)

        if spatial_std < self.settings.spatial_std_threshold:
            if self._low_std_since is None:
                self._low_std_since = now
        else:
            self._low_std_since = None

        if self._baseline_gray is None:
            self._baseline_gray = gray.copy()
        baseline_diff = float(np.mean(np.abs(gray - self._baseline_gray)))
        with self.state.lock:
            self.state.last_baseline_diff = round(baseline_diff, 3)

        if baseline_diff > self.settings.baseline_diff_threshold:
            if self._high_baseline_diff_since is None:
                self._high_baseline_diff_since = now
        else:
            self._high_baseline_diff_since = None

        is_tamper = (
            self._low_std_since is not None
            and now - self._low_std_since >= self.settings.tamper_seconds
        ) or (
            self._high_baseline_diff_since is not None
            and now - self._high_baseline_diff_since >= self.settings.tamper_seconds
        )

        is_frozen = False
        if self._prev_gray is not None:
            diff_score = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
            with self.state.lock:
                self.state.last_diff_score = round(diff_score, 3)

            if diff_score < self.settings.diff_threshold:
                if self._low_diff_since is None:
                    self._low_diff_since = now
                elif now - self._low_diff_since >= self.settings.freeze_seconds:
                    is_frozen = True
            else:
                self._low_diff_since = None

        if is_tamper:
            self._set_status(STATUS_TAMPER)
        elif is_frozen:
            self._set_status(STATUS_FROZEN)
        else:
            self._set_status(STATUS_OK)

        # Referansi yalniza BU karenin kendisi temiz gorunuyorsa guncelle (sustained
        # tamper karari icin gereken bekleme suresine bakmadan, anlik durum). Aksi
        # halde tamper esigine ulasilmadan once gecen "ramp-up" karelerinde (ornegin
        # lens kapatilirken) referans kirlenir ve kurcalanma bitince sistem asla
        # OK durumuna geri donemez (referans hep eski/bozuk kaliyor).
        frame_looks_clean = (
            spatial_std >= self.settings.spatial_std_threshold
            and baseline_diff <= self.settings.baseline_diff_threshold
        )
        if frame_looks_clean:
            self._baseline_gray = 0.95 * self._baseline_gray + 0.05 * gray

        self._prev_gray = gray

    def run(self) -> None:
        while not self._stop_event.is_set():
            capture = self._open_capture()
            if capture is None:
                self._set_status(STATUS_DISCONNECTED)
                self._stop_event.wait(self.settings.reconnect_delay_seconds)
                continue

            self._prev_gray = None
            self._baseline_gray = None
            self._low_diff_since = None
            self._low_std_since = None
            self._high_baseline_diff_since = None

            while not self._stop_event.is_set():
                ret, frame = capture.read()
                if not ret or frame is None:
                    break

                with self.state.lock:
                    self.state.last_frame_time = time.time()

                self._analyze_frame(frame)

                if time.time() - self._last_log_time >= self.settings.log_interval_seconds:
                    self._write_log(self.state.status)

                self._stop_event.wait(self.settings.read_interval_seconds)

            capture.release()
            if not self._stop_event.is_set():
                self._set_status(STATUS_DISCONNECTED)
                self._stop_event.wait(self.settings.reconnect_delay_seconds)


class HealthLogWriter:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._lock = threading.Lock()
        is_new = not csv_path.exists()
        self._file = csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(["timestamp", "camera", "channel", "status", "diff_score"])
            self._file.flush()

    def __call__(self, camera_name: str, channel: int, status: str, diff_score: Optional[float]) -> None:
        with self._lock:
            self._writer.writerow([now_iso(), camera_name, channel, status, diff_score])
            self._file.flush()

    def close(self) -> None:
        self._file.close()


def write_status_json(path: Path, states: dict[str, CameraState]) -> None:
    payload = {name: state.snapshot() for name, state in states.items()}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    try:
        settings = load_settings(args.config)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Ayar hatasi: {exc}", file=sys.stderr)
        return 1

    log_writer = HealthLogWriter(settings.log_csv_path)
    states = {cam.name: CameraState(name=cam.name, channel=cam.channel) for cam in settings.cameras}

    stop_event = threading.Event()

    def handle_signal(signum, frame):
        print("\nDurduruluyor...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    watchers = [
        CameraWatcher(cam, settings, states[cam.name], log_writer, stop_event)
        for cam in settings.cameras
    ]
    for watcher in watchers:
        watcher.start()

    print(f"Watchdog basladi: {len(watchers)} kamera izleniyor.")
    print(f"Durum dosyasi: {settings.status_json_path}")
    print(f"Log dosyasi:   {settings.log_csv_path}")

    try:
        while not stop_event.is_set():
            write_status_json(settings.status_json_path, states)
            summary = ", ".join(f"{name}={state.status}" for name, state in states.items())
            print(f"[{now_iso()}] {summary}")
            stop_event.wait(5.0)
    finally:
        stop_event.set()
        for watcher in watchers:
            watcher.join(timeout=5.0)
        write_status_json(settings.status_json_path, states)
        log_writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
