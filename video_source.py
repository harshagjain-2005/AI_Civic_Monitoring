"""
video_source.py — CivicGuard Video Source Abstraction Layer
============================================================
Provides a unified VideoSource interface that wraps:
  1. Laptop/USB webcam       (OpenCV VideoCapture device index)
  2. Local video file        (OpenCV VideoCapture file path)
  3. Public live stream      (HLS .m3u8 / MJPEG / RTSP URL)

All three source types expose the same open() / read() / release()
interface so the existing dashboard_pro.py detection pipeline can use
any source without modification.

PUBLIC STREAMS — LEGAL NOTICE
------------------------------
Only intentionally public, government-published or open infrastructure
camera feeds are used as demo sources.  No private cameras, no
authentication bypass, no IP scanning.  The demo URLs reference
publicly available streams from US state DOT (Department of
Transportation) infrastructure intended for free public access.

Stream stability note: Government traffic cameras can be taken offline,
change URLs, or have maintenance windows.  The reconnect logic here
handles transient failures gracefully.  If a URL stops working
permanently, replace it in config/cameras.json.

Thread safety
-------------
VideoSource is NOT thread-safe by itself — the caller (dashboard_pro.py)
is responsible for locking around open/read/release as it already does
with _camera_lock for cv2.VideoCapture.
"""

import cv2
import numpy as np
import time
import threading
import os
import subprocess
import sys

# ============================================================
# SOURCE TYPES
# ============================================================

SOURCE_WEBCAM  = "webcam"
SOURCE_FILE    = "file"
SOURCE_STREAM  = "stream"   # HLS / MJPEG / RTSP / any URL


# ============================================================
# RECONNECT CONFIGURATION
# ============================================================

DEFAULT_RECONNECT_INTERVAL = 5.0   # seconds between reconnect attempts
DEFAULT_MAX_RECONNECTS     = 12    # give up after this many consecutive fails (0 = infinite)
DEFAULT_READ_TIMEOUT       = 10.0  # seconds before a stalled read is considered dead
STREAM_FPS_CAP             = 15    # max frames/sec to process from a live stream (avoid overload)


# ============================================================
# FFmpeg stream capture
# ============================================================

class FFmpegVideoCapture:
    """Small OpenCV-compatible capture wrapper for public HLS streams."""

    def __init__(self, uri):
        self._process = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._frame_size = 0
        self._probe(uri)
        if self._frame_size:
            self._process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-nostdin",
                    "-fflags", "+nobuffer",
                    "-allowed_extensions", "ALL",
                    "-reconnect", "1",
                    "-reconnect_streamed", "1",
                    "-reconnect_delay_max", "5",
                    "-i", str(uri),
                    "-an",
                    "-map", "0:v:0",
                    "-f", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )

    def _probe(self, uri):
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-allowed_extensions", "ALL",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "csv=p=0:s=x",
                str(uri),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            check=False,
        )
        metadata = next(
            (line.strip() for line in result.stdout.splitlines() if line.strip()),
            "",
        )
        fields = metadata.split("x")
        if len(fields) < 2:
            return
        self._width = int(fields[0])
        self._height = int(fields[1])
        self._frame_size = self._width * self._height * 3
        if len(fields) > 2 and "/" in fields[2]:
            numerator, denominator = fields[2].split("/", 1)
            if float(denominator) > 0:
                self._fps = float(numerator) / float(denominator)

    def isOpened(self):
        return (
            self._process is not None
            and self._process.poll() is None
            and self._frame_size > 0
        )

    def read(self):
        if not self.isOpened():
            return False, None
        data = bytearray()
        while len(data) < self._frame_size:
            chunk = self._process.stdout.read(self._frame_size - len(data))
            if not chunk:
                return False, None
            data.extend(chunk)
        if len(data) != self._frame_size:
            return False, None
        frame = np.frombuffer(bytes(data), dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        )
        return True, frame

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        if property_id == cv2.CAP_PROP_FPS:
            return self._fps
        return 0.0

    def release(self):
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._process.stdout is not None:
            self._process.stdout.close()
        self._process = None


# ============================================================
# VideoSource
# ============================================================

class VideoSource:
    """
    Unified video source abstraction.

    Usage
    -----
        src = VideoSource(source_type=SOURCE_WEBCAM, source_uri=0)
        src.open()
        while True:
            ok, frame = src.read()
            if not ok:
                break
        src.release()

    Or with context manager:
        with VideoSource(SOURCE_STREAM, "https://example.com/live.m3u8") as src:
            ok, frame = src.read()
    """

    def __init__(
        self,
        source_type: str = SOURCE_WEBCAM,
        source_uri=0,
        reconnect_interval: float = DEFAULT_RECONNECT_INTERVAL,
        max_reconnects: int = DEFAULT_MAX_RECONNECTS,
        fps_cap: int = STREAM_FPS_CAP,
        camera_name: str = "Unknown Source",
    ):
        """
        Parameters
        ----------
        source_type      : SOURCE_WEBCAM | SOURCE_FILE | SOURCE_STREAM
        source_uri       : int (webcam index) | str (file path or URL)
        reconnect_interval : seconds between reconnect attempts (streams only)
        max_reconnects   : 0 = retry forever; N = give up after N attempts
        fps_cap          : Max frames/sec to pull from stream (0 = unlimited)
        camera_name      : Human-readable label for logging / UI
        """
        self.source_type       = source_type
        self.source_uri        = source_uri
        self.reconnect_interval= reconnect_interval
        self.max_reconnects    = max_reconnects
        self.fps_cap           = fps_cap
        self.camera_name       = camera_name

        self._cap              = None
        self._opened           = False
        self._reconnect_count  = 0
        self._last_frame_time  = 0.0
        self._min_frame_gap    = (1.0 / fps_cap) if fps_cap > 0 else 0.0
        self._stream_thread    = None
        self._stream_stop      = threading.Event()
        self._stream_frame     = threading.Event()
        self._stream_lock      = threading.Lock()
        self._latest_frame     = None
        self._stream_read_failed = False

        # Status for the UI
        self.status            = "OFFLINE"   # "OFFLINE" | "CONNECTING" | "LIVE" | "RECONNECTING" | "ERROR"
        self.status_detail     = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """
        Opens the video source.  Returns True on success.
        For live streams, tries up to max_reconnects times if the first
        attempt fails.
        """
        self.status = "CONNECTING"
        self.status_detail = f"Opening {self.camera_name}..."
        return self._connect()

    def read(self):
        """
        Reads the next frame.  Returns (True, frame) on success or
        (False, None) on failure.

        For live streams, a read failure triggers automatic reconnect
        (up to max_reconnects). For webcam/file, returns (False, None)
        immediately.

        Applies fps_cap throttle so the pipeline is not flooded with
        frames faster than needed.
        """
        if not self._opened or self._cap is None:
            return False, None

        # FPS throttle — skip sleeping if fps_cap=0 (unlimited)
        if self._min_frame_gap > 0:
            elapsed = time.time() - self._last_frame_time
            remaining = self._min_frame_gap - elapsed
            if remaining > 0:
                time.sleep(remaining)

        if self.source_type == SOURCE_STREAM:
            if not self._stream_frame.wait(DEFAULT_READ_TIMEOUT):
                return self._handle_stream_failure()

            with self._stream_lock:
                frame = self._latest_frame
                self._latest_frame = None
                self._stream_frame.clear()

            if frame is not None:
                ok = True
            elif self._stream_read_failed:
                return self._handle_stream_failure()
            else:
                ok = False
        else:
            try:
                ok, frame = self._cap.read()
            except Exception as e:
                print(f"[VideoSource] Read exception on '{self.camera_name}': {e}")
                ok, frame = False, None

        self._last_frame_time = time.time()

        if ok and frame is not None:
            self._reconnect_count = 0  # reset on successful read
            self.status = "LIVE"
            self.status_detail = ""
            return True, frame

        # Read failed
        if self.source_type == SOURCE_STREAM:
            return self._handle_stream_failure()
        else:
            self.status = "OFFLINE"
            self.status_detail = "Read failed"
            return False, None

    def release(self):
        """Releases the underlying cv2.VideoCapture and resets state."""
        self._stop_stream_reader()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._opened = False
        self.status = "OFFLINE"
        self.status_detail = ""

    def is_open(self) -> bool:
        return self._opened and self._cap is not None

    def get_info(self) -> dict:
        """Returns metadata about the current source for UI display."""
        info = {
            "type": self.source_type,
            "uri": str(self.source_uri),
            "name": self.camera_name,
            "status": self.status,
            "status_detail": self.status_detail,
            "reconnect_count": self._reconnect_count,
        }
        if self._cap is not None and self._opened:
            try:
                info["width"]  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info["height"] = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info["fps"]    = self._cap.get(cv2.CAP_PROP_FPS)
            except Exception:
                pass
        return info

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stop_stream_reader(self):
        """Stops the stream reader before its capture is released."""
        self._stream_stop.set()
        self._stream_frame.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None

    def _stream_reader(self, cap):
        """Continuously reads a stream while retaining only its newest frame."""
        while not self._stream_stop.is_set():
            try:
                ok, frame = cap.read()
            except Exception as e:
                print(f"[VideoSource] Stream read exception on "
                      f"'{self.camera_name}': {e}")
                ok, frame = False, None

            if not ok or frame is None:
                self._stream_read_failed = True
                self._stream_frame.set()
                return

            with self._stream_lock:
                self._latest_frame = frame
            self._stream_frame.set()

    def _connect(self) -> bool:
        """Creates and opens the underlying cv2.VideoCapture."""
        # Release any existing capture first
        self._stop_stream_reader()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._opened = False

        uri = self.source_uri

        # For streams: set environment hints that help OpenCV's
        # FFmpeg backend handle HLS better
        if self.source_type == SOURCE_STREAM:
            uri = str(uri)

        # MSMF can report a webcam as opened while returning no frames on
        # some Windows camera drivers. Prefer DirectShow, then fall back to
        # the default backend and MSMF.
        backends = [None]
        if self.source_type == SOURCE_WEBCAM and sys.platform == "win32":
            backends = [cv2.CAP_DSHOW, None, cv2.CAP_MSMF]

        cap = None
        selected_backend = None
        last_error = "Source could not be opened"
        if self.source_type == SOURCE_STREAM:
            try:
                candidate = FFmpegVideoCapture(uri)
                if candidate.isOpened():
                    cap = candidate
                    selected_backend = "ffmpeg"
                    print("[VideoSource] Using FFmpeg backend for public HLS stream")
                else:
                    candidate.release()
                    last_error = "FFmpeg could not open the stream"
            except (OSError, ValueError, subprocess.SubprocessError) as e:
                last_error = str(e)
        else:
            for backend in backends:
                try:
                    candidate = (
                        cv2.VideoCapture(uri)
                        if backend is None
                        else cv2.VideoCapture(uri, backend)
                    )
                except Exception as e:
                    last_error = str(e)
                    continue

                if candidate.isOpened():
                    cap = candidate
                    selected_backend = backend
                    break

                candidate.release()

        if cap is None:
            self.status = "ERROR"
            self.status_detail = last_error
            print(f"[VideoSource] '{self.camera_name}' could not be opened "
                  f"(URI: {uri})")
            return False

        # For streams, set a read buffer limit so we don't buffer too many
        # stale frames (reduces latency on slow connections)
        if self.source_type == SOURCE_STREAM:
            self._stream_stop.clear()
            self._stream_frame.clear()
            self._stream_read_failed = False
            with self._stream_lock:
                self._latest_frame = None
            self._stream_thread = threading.Thread(
                target=self._stream_reader,
                args=(cap,),
                daemon=True,
            )
            self._stream_thread.start()

        self._cap     = cap
        self._opened  = True
        self.status   = "LIVE"
        backend_name = {
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "Media Foundation",
        }.get(selected_backend, "default")
        if selected_backend == "ffmpeg":
            backend_name = "FFmpeg"
        self.status_detail = f"Backend: {backend_name}"
        self._reconnect_count = 0
        print(f"[VideoSource] '{self.camera_name}' opened successfully "
              f"(type={self.source_type}, backend={backend_name})")
        return True

    def _handle_stream_failure(self):
        """
        Called when a read() fails on a live stream source.
        Attempts reconnection up to max_reconnects times with
        reconnect_interval delay between each attempt.
        Returns (False, None) if all reconnects fail so the caller
        can shut down cleanly.
        """
        if (self.max_reconnects > 0 and
                self._reconnect_count >= self.max_reconnects):
            print(f"[VideoSource] '{self.camera_name}' — max reconnects "
                  f"({self.max_reconnects}) reached. Giving up.")
            self.status = "ERROR"
            self.status_detail = f"Failed after {self._reconnect_count} reconnect attempts"
            self._opened = False
            return False, None

        self._reconnect_count += 1
        self.status = "RECONNECTING"
        self.status_detail = (f"Attempt {self._reconnect_count}"
                              + (f"/{self.max_reconnects}"
                                 if self.max_reconnects > 0 else ""))
        print(f"[VideoSource] '{self.camera_name}' — stream lost, "
              f"reconnect {self.status_detail} in "
              f"{self.reconnect_interval}s...")

        time.sleep(self.reconnect_interval)

        if self._connect():
            # Try one read immediately after reconnect
            ok, frame = self.read()
            if ok and frame is not None:
                return True, frame

        return False, None


# ============================================================
# FACTORY FUNCTION
# ============================================================

def make_video_source(source_type: str, source_uri=None,
                      camera_name: str = None, fps_cap: int = STREAM_FPS_CAP,
                      reconnect_interval: float = DEFAULT_RECONNECT_INTERVAL,
                      max_reconnects: int = DEFAULT_MAX_RECONNECTS) -> VideoSource:
    """
    Convenience factory.  Applies sensible defaults based on source_type.

    Parameters
    ----------
    source_type  : "webcam" | "file" | "stream"
    source_uri   : None → webcam index 0; int → webcam index;
                   str → file path or URL
    camera_name  : display name; auto-generated if None
    fps_cap      : 0 = no cap; >0 = cap (recommended 10-15 for streams)
    """
    if source_type == SOURCE_WEBCAM:
        idx = int(source_uri) if source_uri is not None else 0
        name = camera_name or f"Webcam [{idx}]"
        return VideoSource(
            source_type=SOURCE_WEBCAM,
            source_uri=idx,
            reconnect_interval=reconnect_interval,
            max_reconnects=0,      # webcam: never give up
            fps_cap=0,             # webcam: no artificial cap
            camera_name=name,
        )

    elif source_type == SOURCE_FILE:
        path = str(source_uri or "")
        name = camera_name or os.path.basename(path) or "Video File"
        return VideoSource(
            source_type=SOURCE_FILE,
            source_uri=path,
            reconnect_interval=reconnect_interval,
            max_reconnects=0,
            fps_cap=0,             # file: no cap, let it run at native FPS
            camera_name=name,
        )

    elif source_type == SOURCE_STREAM:
        url = str(source_uri or "")
        name = camera_name or "Public Live Stream"
        return VideoSource(
            source_type=SOURCE_STREAM,
            source_uri=url,
            reconnect_interval=reconnect_interval,
            max_reconnects=max_reconnects,
            fps_cap=fps_cap,
            camera_name=name,
        )

    else:
        raise ValueError(f"Unknown source_type: {source_type!r}. "
                         f"Must be 'webcam', 'file', or 'stream'.")
