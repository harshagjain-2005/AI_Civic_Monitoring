"""
Smart Civic Monitoring — Canonical Runtime
============================================================
Single entry point for the AI Civic Monitoring System.

    python dashboard_pro.py

This file integrates the detection/verification engine originally
prototyped as a standalone desktop script in main1.py into the Flask
dashboard. The camera loop, tracking, association, temporal
verification, risk scoring, evidence handling and event history logic
are adapted from main1.py; the desktop-only parts of main1.py
(cv2.imshow, cv2.waitKey, the blocking while-loop, module-level camera
open) are NOT used here — camera lifecycle is Flask-driven via
/start and /stop, and frames are streamed via /video_feed.

HONEST SCOPE STATEMENT (see also "Known limitations" in project docs):
This system performs AI-based, context-aware civic violation
verification using visual detection (YOLOv8), object tracking
(ByteTrack), spatial context (person/vehicle-object association,
ground-contact point, perspective-aware ground-plane polygons),
temporal persistence, and evidence-based decision making. It does
NOT perform full 3D scene reconstruction, does NOT recognize human
intent, and does NOT guarantee 100% accuracy or make legal
determinations. Polygons model the ground/restricted area as
projected into this fixed camera's 2D image plane; a real
homography/perspective transform could be added later (see
POLYGON CONFIGURATION section) but is not implemented here.
"""

from flask import Flask, Response, send_from_directory, redirect, render_template, jsonify, request
import cv2
import os
import time
import math
import json
import threading
import tempfile
from collections import deque
from datetime import datetime

import numpy as np

# ============================================================
# VIDEO SOURCE ABSTRACTION (video_source.py)
# ============================================================
# Provides a unified VideoSource that wraps webcam, local video
# file, and public live streams (HLS/MJPEG/RTSP) behind a
# single open/read/release interface, with automatic reconnection
# for stream failures.

try:
    from video_source import (
        VideoSource, make_video_source,
        SOURCE_WEBCAM, SOURCE_FILE, SOURCE_STREAM,
    )
    _VIDEO_SOURCE_AVAILABLE = True
except ImportError as _vs_err:
    _VIDEO_SOURCE_AVAILABLE = False
    SOURCE_WEBCAM = "webcam"
    SOURCE_FILE   = "file"
    SOURCE_STREAM = "stream"
    print(f"[WARN] video_source.py not found: {_vs_err}. "
          f"Multi-source support unavailable; webcam-only mode.")

# Path to the camera presets configuration file
CAMERAS_CONFIG_FILE = os.environ.get(
    "CIVICGUARD_CAMERAS_CONFIG",
    os.path.join("config", "cameras.json")
)

# ============================================================
# OPTIONAL / GUARDED IMPORT: ultralytics
# ============================================================
# Model loading can fail (missing weights, missing GPU/torch issues,
# etc). The dashboard must stay up even if detection is unavailable,
# so this import — and the model loading below — is guarded rather
# than allowed to crash the whole Flask process (requirement: handle
# "model loading failure" without crashing the server).

try:
    from ultralytics import YOLO
    _ULTRALYTICS_IMPORT_ERROR = None
except Exception as _e:                                   # pragma: no cover
    YOLO = None
    _ULTRALYTICS_IMPORT_ERROR = str(_e)


app = Flask(__name__)


# ============================================================
# CONFIGURATION — CONFIDENCE THRESHOLDS
# ============================================================
# Each model/class gets its own explicit threshold. Previously
# (main1.py) VEHICLE_CONFIDENCE was defined but never actually
# applied — vehicles were filtered using the same coarse call-level
# threshold as everything else. Fixed here: the model.track() call
# uses the *minimum* of the two COCO thresholds so nothing is
# dropped too early, and then each detection is re-checked against
# its own class-specific threshold in the detection loop below
# (see run_person_vehicle_detection).

GARBAGE_CONFIDENCE = 0.60
PERSON_CONFIDENCE = 0.55
VEHICLE_CONFIDENCE = 0.55

_COCO_CALL_CONFIDENCE = min(PERSON_CONFIDENCE, VEHICLE_CONFIDENCE)

VEHICLE_LABELS = ("car", "motorcycle", "bus", "truck")
NOPARK_DEBUG = os.environ.get("CIVICGUARD_NOPARK_DEBUG", "").lower() in (
    "1", "true", "yes", "on"
)

# ============================================================
# CONFIGURATION — TEMPORAL VERIFICATION
# ============================================================
# All thresholds are centralized here (not scattered through the
# code) so they're easy to tune for a real deployment/camera.

MIN_ASSOCIATION_FRAMES = 5        # littering: frames person must be near garbage
MIN_SEPARATION_FRAMES = 5         # littering: frames person must be away after being near
MIN_STATIONARY_FRAMES = 8         # littering: frames garbage must stay still
MIN_GARBAGE_PERSISTENCE = 2.0     # littering: seconds garbage must remain after separation
VERIFY_FRAMES = 10                # trespass/no-parking: consecutive frames inside polygon

PERSON_GARBAGE_DISTANCE_RATIO = 0.18   # association distance, normalized to frame width
SEPARATION_DISTANCE_RATIO = 0.25       # separation distance, normalized to frame width
STATIONARY_MOVEMENT_PIXELS = 25        # max frame-to-frame garbage movement to count as "stationary"
STATIONARY_WINDOW_FRAMES = 10          # recent footpoint window used to absorb tracker jitter

EVIDENCE_COOLDOWN = 5              # seconds between evidence saves for the same continuous event

HIGH_RISK_THRESHOLD = 70           # risk >= this value counts toward the "High-Risk Events" KPI

TRACK_GRACE_FRAMES = 8             # frames a tracked person/vehicle may vanish (occlusion/track-ID
                                    # flicker) before its zone-persistence state is reset to "left"

# ============================================================
# CONFIGURATION — PERSPECTIVE-AWARE GROUND POLYGONS
# ============================================================
# These replace the old axis-aligned rectangles with polygons, so a
# restricted ground area that appears as a non-rectangular shape in
# a real CCTV camera's perspective (e.g. a trapezoid because the far
# edge of a lot is smaller in the image than the near edge) can be
# modeled directly.
#
# HOW TO ADJUST FOR YOUR CAMERA:
#   Easiest: use the in-dashboard "Zone Calibration" panel (sidebar)
#   — it shows a live snapshot from your camera, you click points
#   directly on the restricted floor area, and it saves the polygon
#   for you (persisted to zones_config.json, survives restarts).
#   Manual alternative: edit the points below directly; go around the
#   shape in order (clockwise or counter-clockwise both work with
#   cv2.pointPolygonTest as long as the points aren't crossed).
#
# The defaults below are intentionally NOT a generic centered
# rectangle — they're trapezoids (narrower at the top of the frame,
# wider at the bottom), because a real ground-plane restricted area
# viewed by a CCTV camera converges toward the horizon/vanishing
# point rather than staying a fixed-width box. They are still only
# illustrative placeholders; calibrate them to your real camera via
# the Zone Calibration panel or by editing zones_config.json.
#
# This is 2D image-plane polygon modeling, not 3D reconstruction —
# see the module docstring's honest scope statement. The structure
# (a fixed polygon in image pixel-space) is what a future
# homography/perspective transform would replace/extend; no such
# transform is implemented here.

TRESPASS_ZONE = np.array([
    [60, 140],
    [260, 140],
    [300, 420],
    [20, 420],
], dtype=np.int32)

NO_PARKING_ZONE = np.array([
    [380, 150],
    [580, 150],
    [620, 420],
    [340, 420],
], dtype=np.int32)

ZONES_CONFIG_FILE = os.environ.get("CIVICGUARD_ZONES_CONFIG", "zones_config.json")
EVIDENCE_DIR = os.environ.get("CIVICGUARD_EVIDENCE_DIR", "evidence")
LOGS_DIR = os.environ.get("CIVICGUARD_LOGS_DIR", "logs")
HISTORY_FILE = os.environ.get("CIVICGUARD_HISTORY_FILE", os.path.join(LOGS_DIR, "violation_history.json"))
ALERTS_FILE = os.environ.get("CIVICGUARD_ALERTS_FILE", os.path.join(LOGS_DIR, "alerts.txt"))

ALLOWED_EVIDENCE_CATEGORIES = {"garbage", "trespassing", "noparking"}


def atomic_json_dump(filepath, data, indent=2):
    """Atomically writes JSON data to a file using a temp file in the same
    directory and os.replace. This guarantees that readers never see a
    partially-written or truncated file, and on POSIX/Windows (Python 3.3+),
    os.replace is an atomic rename."""
    dirname = os.path.dirname(filepath) or "."
    os.makedirs(dirname, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=dirname, prefix=os.path.basename(filepath) + ".tmp_")
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, filepath)
        return True
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise e


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(os.path.join(EVIDENCE_DIR, "garbage"), exist_ok=True)
os.makedirs(os.path.join(EVIDENCE_DIR, "trespassing"), exist_ok=True)
os.makedirs(os.path.join(EVIDENCE_DIR, "noparking"), exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ============================================================
# ZONE CALIBRATION — persistence for operator-defined polygons
# ============================================================
# Lets the operator define the ACTUAL restricted floor area for
# their camera (via the Zone Calibration panel, or by hand-editing
# this file) instead of being stuck with the generic default
# trapezoids above. Saved to zones_config.json so it survives
# restarts. Never touches logs/evidence — a completely separate file.

def load_zones_config():
    """Loads a saved zone calibration, if any. Never raises — a
    missing or corrupt file just means the module-level defaults
    (TRESPASS_ZONE / NO_PARKING_ZONE) stay in effect."""

    if not os.path.exists(ZONES_CONFIG_FILE):
        return None

    try:
        with open(ZONES_CONFIG_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        result = {}

        for key in ("trespass_zone", "no_parking_zone"):
            pts = data.get(key)
            if isinstance(pts, list) and len(pts) >= 3:
                result[key] = np.array(
                    [[int(p[0]), int(p[1])] for p in pts], dtype=np.int32
                )

        return result or None

    except Exception as e:
        print(f"[WARN] {ZONES_CONFIG_FILE} could not be read ({e}); "
              f"using default zone polygons instead.")
        return None


def save_zones_config():
    try:
        atomic_json_dump(ZONES_CONFIG_FILE, {
            "trespass_zone": TRESPASS_ZONE.tolist(),
            "no_parking_zone": NO_PARKING_ZONE.tolist(),
        }, indent=2)
        return True
    except Exception as e:
        print(f"[WARN] failed to save {ZONES_CONFIG_FILE}: {e}")
        return False


_saved_zones = load_zones_config()
if _saved_zones:
    if "trespass_zone" in _saved_zones:
        TRESPASS_ZONE = _saved_zones["trespass_zone"]
    if "no_parking_zone" in _saved_zones:
        NO_PARKING_ZONE = _saved_zones["no_parking_zone"]
    print(f"[INFO] Loaded calibrated zone polygons from {ZONES_CONFIG_FILE}.")


# ============================================================
# STRUCTURED EVENT HISTORY (logs/violation_history.json)
# ============================================================
# This is the primary, structured source of truth for VERIFIED
# events — used for KPI counters, the Recent Violations table, the
# Active Event panel, Analytics charts, and the Evidence gallery.
# logs/alerts.txt (below) remains a parallel human-readable
# audit trail, written on the same events, but is not re-parsed for
# the dashboard UI (avoids the fragile substring-matching approach
# used previously, which breaks the moment log wording changes).

_history_lock = threading.Lock()

VIOLATION_TO_CATEGORY = {
    "LITTERING": "garbage",
    "TRESPASSING": "trespassing",
    "NO PARKING": "noparking",
}

CATEGORY_LABEL = {
    "garbage": "LITTERING",
    "trespassing": "TRESPASSING",
    "noparking": "NO PARKING",
}


def normalize_event_record(record):
    """Normalizes any legacy or modern event record in-memory.
    Ensures backward compatibility for older records that used 'type'
    instead of 'violation', or lacked 'category', 'persistence', etc.
    Does not modify or rewrite the persistent JSON file on disk."""
    if not isinstance(record, dict):
        return {}

    ev = dict(record)

    # 1. Normalize violation name and category
    raw_violation = ev.get("violation")
    raw_type = ev.get("type")
    raw_category = ev.get("category")

    if raw_violation:
        violation = str(raw_violation).upper()
        if violation in ("LITTERING", "GARBAGE"):
            violation = "LITTERING"
            category = "garbage"
        elif violation in ("TRESPASSING", "TRESPASS"):
            violation = "TRESPASSING"
            category = "trespassing"
        elif violation in ("NO PARKING", "NOPARKING", "PARKING"):
            violation = "NO PARKING"
            category = "noparking"
        else:
            category = raw_category or "garbage"
    elif raw_type:
        type_lower = str(raw_type).lower()
        if type_lower in ("garbage", "littering"):
            violation = "LITTERING"
            category = "garbage"
        elif type_lower in ("trespassing", "trespass"):
            violation = "TRESPASSING"
            category = "trespassing"
        elif type_lower in ("noparking", "no parking"):
            violation = "NO PARKING"
            category = "noparking"
        else:
            violation = type_lower.upper()
            category = type_lower
    elif raw_category:
        cat_lower = str(raw_category).lower()
        category = cat_lower
        violation = "LITTERING" if cat_lower == "garbage" else ("TRESPASSING" if cat_lower == "trespassing" else ("NO PARKING" if cat_lower == "noparking" else cat_lower.upper()))
    else:
        violation = "UNKNOWN"
        category = "garbage"

    ev["violation"] = violation
    ev["category"] = category

    # 2. Normalize persistence
    if "persistence" not in ev:
        ev["persistence"] = 0

    # 3. Normalize confidence and risk
    if "confidence" in ev:
        try:
            ev["confidence"] = float(ev["confidence"])
        except (ValueError, TypeError):
            ev["confidence"] = 0.0
    else:
        ev["confidence"] = 0.0

    if "risk" in ev:
        try:
            ev["risk"] = int(ev["risk"])
        except (ValueError, TypeError):
            ev["risk"] = 0
    else:
        ev["risk"] = 0

    # 4. Normalize evidence path and filename safely for Windows & POSIX
    if ev.get("evidence"):
        ev_path = str(ev["evidence"]).replace("\\", "/")
        filename = os.path.basename(ev_path)
        ev["evidence_filename"] = filename
        ev["evidence_url"] = f"/evidence/{category}/{filename}"
    else:
        ev["evidence_filename"] = None
        ev["evidence_url"] = None

    return ev


def load_history():
    """Loads violation_history.json safely and normalizes records in memory.
    Never raises — a missing or corrupt file simply starts a fresh (and, for
    corrupt files, backed-up) history rather than crashing the server."""

    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, list):
                return [normalize_event_record(item) for item in data]
            return []
    except Exception as e:
        # Corrupt JSON: preserve the broken file for inspection
        # instead of silently discarding it, then start fresh.
        try:
            corrupt_path = HISTORY_FILE + f".corrupt-{int(time.time())}"
            os.replace(HISTORY_FILE, corrupt_path)
            print(f"[WARN] violation_history.json was corrupt ({e}); "
                  f"backed up to {corrupt_path} and starting a fresh history.")
        except Exception:
            pass
        return []


violation_history = load_history()


def _backfill_missing_ids(history):
    """Backward-compatibility fix: older logs/violation_history.json
    files (written before the 'id' field existed) may contain records
    with no 'id' key at all. Rather than crashing or discarding that
    history, every record keeps its existing 'id' if it has one, and
    any record missing a valid integer 'id' is assigned the next
    unused one, in file order. Returns the id to use for the next new
    event, and whether anything changed (so callers can decide
    whether to persist the backfill)."""

    max_existing_id = 0
    for record in history:
        record_id = record.get("id")
        if isinstance(record_id, int) and record_id > max_existing_id:
            max_existing_id = record_id

    next_id = max_existing_id + 1
    changed = False

    for record in history:
        record_id = record.get("id")
        if not isinstance(record_id, int):
            record["id"] = next_id
            next_id += 1
            changed = True

    return next_id, changed


_next_event_id, _history_backfilled = _backfill_missing_ids(violation_history)

if _history_backfilled:
    try:
        atomic_json_dump(HISTORY_FILE, violation_history, indent=2)
        print(f"[INFO] {HISTORY_FILE} contained records without an "
              f"'id' field (from an older version). Assigned IDs "
              f"{1}..{_next_event_id - 1} in file order and saved. "
              f"No records were removed.")
    except Exception as _e:
        print(f"[WARN] could not persist backfilled IDs to "
              f"{HISTORY_FILE}: {_e}. Continuing with in-memory IDs "
              f"for this session.")


def save_history(event_without_id):
    """Appends one verified event to the in-memory + on-disk history.
    Assigns a monotonically increasing 'id' (continues correctly
    across restarts since it's derived from the loaded history) so
    the frontend can detect 'this is a new event' without guessing
    from timestamps alone. Uses atomic write to prevent corruption."""

    global _next_event_id

    with _history_lock:
        event = dict(event_without_id)
        event["id"] = _next_event_id
        _next_event_id += 1

        normalized_event = normalize_event_record(event)
        violation_history.append(normalized_event)

        try:
            atomic_json_dump(HISTORY_FILE, violation_history, indent=2)
        except Exception as e:
            print(f"[WARN] failed to write {HISTORY_FILE}: {e}")

        return normalized_event


# ============================================================
# ALERT LOG (logs/alerts.txt) — human-readable audit trail
# ============================================================
# Rich format (new):
#   [ts] LITTERING VERIFIED | confidence=0.87 | risk=86 | evidence=path | association=0.82 | persistence=18
#   [ts] TRESPASSING VERIFIED | confidence=0.89 | risk=82 | evidence=path | persistence=14
#   [ts] NO PARKING VERIFIED | confidence=0.91 | risk=84 | evidence=path | persistence=20
#
# Legacy format (old dashboard.py / dashboard_live.py / earlier
# dashboard_pro.py) lines such as "[ts] GARBAGE DETECTED - bottle"
# may still exist at the top of an existing alerts.txt from before
# this integration. They are never deleted or rewritten — new lines
# are only appended — so no history is lost. The dashboard UI itself
# now reads violation_history.json rather than parsing this file, so
# format differences between old and new lines cannot break the UI.

def log_alert(violation, confidence, risk, evidence, extra=""):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = (
        f"[{timestamp}] {violation} VERIFIED | "
        f"confidence={confidence:.2f} | "
        f"risk={risk} | "
        f"evidence={evidence}"
    )

    if extra:
        message += " | " + extra

    print(message)

    try:
        with open(ALERTS_FILE, "a", encoding="utf-8") as file:
            file.write(message + "\n")
    except Exception as e:
        print(f"[WARN] failed to write {ALERTS_FILE}: {e}")

    return timestamp


# ============================================================
# RISK SCORE
# ============================================================
# score = confidence*45 + min(persistence, 60)*0.5 + context_score*35
# risk  = min(int(score), 100)
#
# Design notes (documented per requirement):
#  - confidence in [0,1]: contributes up to 45 points. Detection
#    quality matters most, but alone cannot reach "high risk".
#  - persistence (raw frame/stability count) is capped at 60 BEFORE
#    weighting, then weighted by 0.5 -> contributes at most 30
#    points. This means persistence alone (confidence=0, context=0)
#    can never exceed 30/100 — satisfies "do not allow persistence
#    alone to trivially force risk to 100".
#  - context_score in [0,1] (association strength for littering, or
#    a fixed 1.0 for confirmed zone-entry events) contributes up to
#    35 points.
#  - Maximum possible raw score is 45+30+35=110, clipped to 100.

def calculate_risk(confidence, persistence, context_score=0.0):

    score = (
        confidence * 45
        + min(persistence, 60) * 0.5
        + context_score * 35
    )

    return min(int(score), 100)


# ============================================================
# EVIDENCE QUALITY
# ============================================================
# A lightweight, genuinely-computed (not fabricated) quality score
# for the single frame captured at verification time. Combines:
#   - clarity: Laplacian variance of the cropped detection region,
#     normalized against CLARITY_NORM (a blurry/motion-smeared crop
#     has low variance; a crisp one has high variance). This uses
#     only cv2, already a dependency — no extra libraries needed.
#   - confidence: the model's own detection confidence.
#   - context_score: the same context signal used in risk scoring
#     (association strength for littering, 1.0 for zone-entry
#     events).
#
# NOTE (honest limitation): this scores the ONE frame that triggered
# the event; it does not buffer and compare multiple candidate
# frames to pick the "best" one. True multi-frame evidence selection
# is not implemented — see "Known limitations" in the project docs.

CLARITY_NORM = 400.0


def compute_evidence_quality(frame, bbox, confidence, context_score=0.0):

    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            clarity_score = 0.0
        else:
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            clarity_score = min(lap_var / CLARITY_NORM, 1.0)

    except Exception:
        clarity_score = 0.0

    quality = 0.4 * float(clarity_score) + 0.4 * float(confidence) + 0.2 * float(context_score)

    return round(float(min(max(quality, 0.0), 1.0)), 3)


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def point_to_bbox_distance(point, bbox):
    """Distance from a point to the NEAREST edge/corner of a bounding
    box (0 if the point is inside the box). This is used for
    person-garbage association instead of plain center-to-center
    distance.

    Why this matters: a person's bounding-box CENTER is roughly
    torso height. A dropped or held object is usually near waist/hand
    height, closer to the box's lower half — so on a tall person, the
    true object-to-person gap can be much smaller than the
    center-to-center distance suggests, causing missed associations
    even when the object is clearly right next to the person. Using
    distance-to-nearest-edge instead fixes that failure mode directly
    (better geometry), WITHOUT touching PERSON_GARBAGE_DISTANCE_RATIO
    or any other threshold — so this makes association more reliable
    without loosening what counts as "close"."""

    px, py = point
    x1, y1, x2, y2 = bbox

    nearest_x = min(max(px, x1), x2)
    nearest_y = min(max(py, y1), y2)

    return distance((px, py), (nearest_x, nearest_y))


def person_ground_point(bbox):
    """Approximate ground-contact point for a person: bottom-center
    of the bounding box, NOT the box center. A standing person's feet
    (not their torso/center) are what actually occupies ground-plane
    position, so this point is what gets tested against the
    perspective-aware restricted-area polygon."""

    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, y2)


def vehicle_ground_point(bbox):
    """Same ground-contact rationale as person_ground_point: a
    vehicle's footprint on the ground is best approximated by the
    bottom-center of its bounding box, not the box's visual center."""

    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, y2)


def point_in_zone(point, zone_polygon):
    """Point-in-polygon test using OpenCV. Returns True if `point`
    is inside or on the boundary of `zone_polygon`."""

    result = cv2.pointPolygonTest(
        zone_polygon,
        (float(point[0]), float(point[1])),
        False,
    )
    return result >= 0


# ============================================================
# GARBAGE TRACK STATE — LITTERING VERIFICATION STATE MACHINE
# ============================================================
# Ported from main1.py. A detected garbage object is NEVER by itself
# a verified violation. It must pass through:
#   STAGE 1  Association       (a person is near the garbage)
#   STAGE 2  Separation        (that person then moves away)
#   STAGE 3  Stability         (the garbage stops moving)
#   STAGE 4  Post-separation   (enough real time has passed since
#            persistence       separation, with stability holding)
#   STAGE 5  Final verification -> LITTERING VERIFIED
#
# Intermediate internal states (not shown as verified alerts):
#   ASSOCIATING / SEPARATION DETECTED / STABILIZING / VERIFYING

garbage_tracks = {}


def create_garbage_track(track_id, center, confidence):

    garbage_tracks[track_id] = {
        "center_history": deque(maxlen=30),
        "confidence_history": deque(maxlen=30),
        "association_frames": 0,
        "separation_frames": 0,
        "stationary_frames": 0,
        "person_near": False,
        "person_left": False,
        "associated_person": None,
        "first_seen": time.time(),
        "separation_time": None,
        "verified": False,
        "last_event_time": 0,
        "last_center": center,
    }

    garbage_tracks[track_id]["center_history"].append(center)
    garbage_tracks[track_id]["confidence_history"].append(confidence)


def garbage_track_stage(state):
    """Returns a human-readable intermediate stage label for display
    purposes only. This label is informational — it is NEVER treated
    as a verified violation; only state["verified"] == True is."""

    if state["verified"]:
        return "VERIFIED"
    if state["separation_frames"] > 0 and state["stationary_frames"] >= MIN_STATIONARY_FRAMES:
        return "VERIFYING"
    if state["separation_frames"] > 0 and state["stationary_frames"] > 0:
        return "STABILIZING"
    if state["person_left"]:
        return "SEPARATION DETECTED"
    if state["association_frames"] > 0:
        return "ASSOCIATING"
    return "DETECTED"


def process_garbage_relation(track_id, garbage_center, garbage_confidence, persons, frame_width):
    """Updates one garbage track's state machine for the current
    frame. Returns the track's state dict (verified flag, frame
    counters, etc.) for the caller to act on."""

    if track_id not in garbage_tracks:
        create_garbage_track(track_id, garbage_center, garbage_confidence)

    state = garbage_tracks[track_id]

    state["center_history"].append(garbage_center)
    state["confidence_history"].append(garbage_confidence)

    # -------------------- FIND CLOSEST PERSON --------------------
    # Uses distance from the garbage center to the NEAREST EDGE of
    # each person's bounding box (point_to_bbox_distance), not
    # center-to-center distance. See point_to_bbox_distance's
    # docstring: this fixes missed associations on tall/close people
    # without loosening the distance threshold itself. Falls back to
    # center-to-center only if a person dict has no "bbox" (keeps
    # this function safe to call with older/minimal person records).

    closest_person = None
    closest_distance = float("inf")

    for person in persons:
        if "bbox" in person:
            d = point_to_bbox_distance(garbage_center, person["bbox"])
        else:
            d = distance(garbage_center, person["center"])
        if d < closest_distance:
            closest_distance = d
            closest_person = person

    association_distance = frame_width * PERSON_GARBAGE_DISTANCE_RATIO
    separation_distance = frame_width * SEPARATION_DISTANCE_RATIO

    # -------------------- STAGE 1: ASSOCIATION --------------------

    if closest_person is not None and closest_distance <= association_distance:
        state["association_frames"] += 1
        state["person_near"] = True
        state["associated_person"] = closest_person["id"]
    else:
        # Do not immediately erase association — separation requires
        # the person to have first been near, then move clearly away.
        if state["person_near"]:
            if closest_person is None or closest_distance >= separation_distance:
                state["separation_frames"] += 1
                state["person_left"] = True
                if state["separation_time"] is None:
                    state["separation_time"] = time.time()

    # -------------------- STAGE 3: STABILITY --------------------
    # (garbage must stop moving; large jumps reset the counter)

    last_center = state["last_center"]
    movement = distance(garbage_center, last_center)

    if movement <= STATIONARY_MOVEMENT_PIXELS:
        state["stationary_frames"] += 1
    else:
        state["stationary_frames"] = 0

    state["last_center"] = garbage_center

    # -------------------- STAGE 4/5: FINAL VERIFICATION --------------------

    if (
        state["association_frames"] >= MIN_ASSOCIATION_FRAMES
        and state["separation_frames"] >= MIN_SEPARATION_FRAMES
        and state["stationary_frames"] >= MIN_STATIONARY_FRAMES
        and state["separation_time"] is not None
        and (time.time() - state["separation_time"]) >= MIN_GARBAGE_PERSISTENCE
    ):
        state["littering_verified_this_frame"] = not state["verified"]
        state["verified"] = True
    else:
        state["littering_verified_this_frame"] = False

    return state


# ============================================================
# TRESPASS / NO-PARKING PER-TRACK ZONE STATE
# ============================================================
# Unlike a single global frame counter (which would conflate
# unrelated people/vehicles passing through at different times),
# each tracked person/vehicle ID gets its own persistence counter.
# This also enables the required reset-on-exit behavior: leaving the
# polygon resets frames_inside, so a person who steps out and back in
# starts a fresh verification window rather than picking up where a
# much earlier, unrelated visit left off.

trespass_tracks = {}
nopark_tracks = {}


def update_zone_track(
    track_state,
    track_id,
    inside,
    confidence,
    evidence_cooldown_ok,
    reference_point=None,
    require_stationary=False,
):
    """Advances one tracked object's zone-persistence state machine
    for the current frame. Returns (verified_now, entry) where
    verified_now is True only on the single frame verification is
    first reached for this continuous stay (duplicate-prevention).

    When requested, ``reference_point`` is tracked across frames and
    verification additionally requires the object to remain within the
    configured movement threshold. This is used for No Parking so a
    moving vehicle passing through the zone is not treated as parked.
    """

    if track_id not in track_state:
        track_state[track_id] = {
            "frames_inside": 0,
            "missed_frames": 0,
            "verified": False,
            "confidence_history": deque(maxlen=30),
            "position_history": deque(maxlen=30),
            "stationary_frames": 0,
            "last_movement": None,
            "last_event_time": 0,
        }

    entry = track_state[track_id]

    if inside:
        entry["frames_inside"] += 1
        entry["missed_frames"] = 0
        entry["confidence_history"].append(confidence)

        if reference_point is not None:
            entry["position_history"].append(reference_point)
            if len(entry["position_history"]) > 1:
                entry["last_movement"] = distance(
                    reference_point, entry["position_history"][-2]
                )

            recent_positions = list(entry["position_history"])[-STATIONARY_WINDOW_FRAMES:]
            recent_array = np.array(recent_positions, dtype=np.float32)
            anchor = (
                recent_array.min(axis=0) + recent_array.max(axis=0)
            ) / 2.0
            envelope_radius = max(
                distance(tuple(position), tuple(anchor))
                for position in recent_positions
            )
            if envelope_radius <= STATIONARY_MOVEMENT_PIXELS:
                entry["stationary_frames"] += 1
            else:
                entry["stationary_frames"] = 0
    else:
        entry["frames_inside"] = 0
        entry["verified"] = False
        entry["stationary_frames"] = 0
        entry["last_movement"] = None
        entry["position_history"].clear()

    verified_now = False

    stationary_ok = (
        not require_stationary
        or entry["stationary_frames"] >= MIN_STATIONARY_FRAMES
    )

    if (
        entry["frames_inside"] >= VERIFY_FRAMES
        and stationary_ok
        and not entry["verified"]
    ):
        entry["verified"] = True
        verified_now = True

    return verified_now, entry


def decay_missing_tracks(track_state, seen_ids):
    """Called once per frame with the set of track IDs actually
    detected this frame. Any tracked object absent this frame is
    given a short grace period (TRACK_GRACE_FRAMES) to survive brief
    occlusion / track-ID flicker before its zone-persistence state is
    reset, as if it had left the polygon."""

    for track_id, entry in list(track_state.items()):
        if track_id in seen_ids:
            continue

        entry["missed_frames"] += 1

        if entry["missed_frames"] > TRACK_GRACE_FRAMES:
            entry["frames_inside"] = 0
            entry["verified"] = False
            entry["stationary_frames"] = 0
            entry["last_movement"] = None
            entry["position_history"].clear()


def log_nopark_debug(vehicle, inside, entry, verified_now, reason=None):
    if not NOPARK_DEBUG:
        return

    if reason is None:
        if not inside:
            reason = "footpoint_outside_zone"
        elif entry["frames_inside"] < VERIFY_FRAMES:
            reason = "dwell_not_reached"
        elif entry["stationary_frames"] < MIN_STATIONARY_FRAMES:
            reason = "stationary_threshold_not_reached"
        elif not verified_now:
            reason = "already_verified"
        else:
            reason = "verified"

    print(
        "[NOPARK] "
        f"track_id={vehicle['id']} source={vehicle.get('track_source', 'unknown')} "
        f"class={vehicle['label']} "
        f"confidence={vehicle['confidence']:.3f} bbox={vehicle['bbox']} "
        f"footpoint={vehicle['ground_point']} inside={inside} "
        f"movement={entry.get('last_movement')} "
        f"stationary_frames={entry['stationary_frames']}/"
        f"{MIN_STATIONARY_FRAMES} "
        f"dwell_frames={entry['frames_inside']}/{VERIFY_FRAMES} "
        f"verified={entry['verified']} verified_now={verified_now} "
        f"reason={reason}"
    )


# ============================================================
# MODEL LOADING (guarded — must not crash the Flask process)
# ============================================================

garbage_model = None
coco_model = None
models_loaded = False
model_load_error = None

if YOLO is None:
    model_load_error = f"ultralytics import failed: {_ULTRALYTICS_IMPORT_ERROR}"
    print(f"[WARN] {model_load_error}. Detection will be unavailable "
          f"until this is resolved (install ultralytics).")
else:
    try:
        print("Loading garbage detection model...")
        garbage_model = YOLO("runs/detect/train/weights/best.pt")

        print("Loading YOLOv8 COCO model...")
        coco_model = YOLO("yolov8n.pt")

        models_loaded = True
        print("Models loaded successfully.")

    except Exception as e:
        model_load_error = str(e)
        garbage_model = None
        coco_model = None
        models_loaded = False
        print(f"[WARN] Model loading failed: {model_load_error}. "
              f"The dashboard will still run; Live Monitoring / "
              f"detection will be unavailable until model files are "
              f"present at the expected paths.")


# ============================================================
# CAMERA / RUNTIME STATE
# ============================================================

camera_running = False
cap = None                  # kept as alias; points to video_source_obj._cap
_camera_lock = threading.Lock()

# --- Video Source abstraction state ---
video_source_obj = None                 # VideoSource instance or None
current_source_type  = SOURCE_WEBCAM   # "webcam" | "file" | "stream"
current_source_label = "Webcam"        # human-readable name for UI
current_source_url   = ""              # URL or path (empty for webcam 0)
_live_processing_thread = None
_live_processing_stop = threading.Event()
_live_frame_event = threading.Event()
_live_frame_lock = threading.Lock()
_live_input_frame = None
_live_output_frame = None
_latest_camera_frame = None

last_alert = "No Active Alerts"
# Active events are session-scoped; persisted history remains available
# in the archive but must not populate the live monitoring panel.
active_event = None

last_garbage_save = 0
last_trespass_save = 0
last_nopark_save = 0


def _reset_tracking_state():
    """Clears all per-track state. Called on /start so each camera
    session begins with a clean verification state rather than
    carrying over stale associations/counters from a previous run."""

    global last_garbage_save, last_trespass_save, last_nopark_save

    garbage_tracks.clear()
    trespass_tracks.clear()
    nopark_tracks.clear()
    last_garbage_save = 0
    last_trespass_save = 0
    last_nopark_save = 0


def _live_processing_loop():
    """Processes only the newest pending stream frame."""
    global _live_input_frame, _live_output_frame

    while not _live_processing_stop.is_set():
        if not _live_frame_event.wait(0.5):
            continue

        with _live_frame_lock:
            frame = _live_input_frame
            _live_input_frame = None
            _live_frame_event.clear()

        if frame is None:
            continue

        try:
            processed = process_frame(frame)
        except Exception as e:
            print(f"[WARN] live frame processing error: {e}")
            processed = frame

        with _live_frame_lock:
            _live_output_frame = processed


def _start_live_processing():
    """Starts the single stream-only detection worker."""
    global _live_processing_thread, _live_input_frame, _live_output_frame

    _live_processing_stop.clear()
    _live_frame_event.clear()
    with _live_frame_lock:
        _live_input_frame = None
        _live_output_frame = None
    _live_processing_thread = threading.Thread(
        target=_live_processing_loop,
        name="live-frame-processing",
        daemon=True,
    )
    _live_processing_thread.start()


def _stop_live_processing():
    """Stops the stream-only detection worker and clears its frames."""
    global _live_processing_thread, _live_input_frame, _live_output_frame

    _live_processing_stop.set()
    _live_frame_event.set()
    if _live_processing_thread is not None:
        _live_processing_thread.join()
        _live_processing_thread = None
    with _live_frame_lock:
        _live_input_frame = None
        _live_output_frame = None
    _live_frame_event.clear()


def _submit_live_frame(frame):
    """Replaces any pending stream frame instead of queueing a backlog."""
    global _live_input_frame

    with _live_frame_lock:
        _live_input_frame = frame
        _live_frame_event.set()


def _latest_live_output():
    with _live_frame_lock:
        return _live_output_frame


# ============================================================
# CAMERAS CONFIG LOADER
# ============================================================

def load_cameras_config():
    """Loads config/cameras.json for the public stream presets list."""
    try:
        if os.path.exists(CAMERAS_CONFIG_FILE):
            with open(CAMERAS_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {CAMERAS_CONFIG_FILE}: {e}")
    return {"demo_streams": [], "default_fps_cap": 10,
            "default_reconnect_interval": 5, "default_max_reconnects": 20}


# ============================================================
# DETECTION HELPERS
# ============================================================

def run_garbage_detection(frame):
    """Runs the custom garbage model with ByteTrack. Returns a list
    of dicts: {id, bbox, center, confidence}. Falls back to a
    synthetic ID (position-derived) only if tracking briefly fails to
    assign one, matching main1.py's fallback behavior."""

    detections = []

    if garbage_model is None:
        return detections

    results = garbage_model.track(
        frame,
        conf=GARBAGE_CONFIDENCE,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < GARBAGE_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = center[0] * 10000 + center[1]

            detections.append({
                "id": track_id,
                "bbox": (x1, y1, x2, y2),
                "center": center,
                "confidence": confidence,
            })

    return detections


def run_person_vehicle_detection(frame):
    """Runs the COCO model with ByteTrack for persons and vehicles.
    Applies the coarse call-level threshold (min of the two
    thresholds) and then re-filters each detection against its own
    class-specific threshold — this is the fix for the main1.py issue
    where VEHICLE_CONFIDENCE was defined but never actually applied.

    Returns (persons, vehicles), each a list of dicts:
    {id, bbox, center, ground_point, confidence, label}."""

    persons, vehicles = [], []

    if coco_model is None:
        return persons, vehicles

    results = coco_model.track(
        frame,
        conf=_COCO_CALL_CONFIDENCE,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            label = coco_model.names[cls]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            track_id = int(box.id[0]) if box.id is not None else None

            if label == "person":
                if confidence < PERSON_CONFIDENCE:
                    continue
                persons.append({
                    "id": track_id if track_id is not None else center[0] * 10000 + center[1],
                    "bbox": bbox,
                    "center": center,
                    "ground_point": person_ground_point(bbox),
                    "confidence": confidence,
                    "label": label,
                })

            elif label in VEHICLE_LABELS:
                if NOPARK_DEBUG:
                    print(
                        "[NOPARK] detection "
                        f"track_id={track_id} class={label} "
                        f"confidence={confidence:.3f} bbox={bbox}"
                    )
                if confidence < VEHICLE_CONFIDENCE:
                    if NOPARK_DEBUG:
                        print(
                            "[NOPARK] rejected reason=vehicle_confidence_below_threshold "
                            f"required={VEHICLE_CONFIDENCE:.3f}"
                        )
                    continue
                vehicles.append({
                    "id": track_id if track_id is not None else center[0] * 10000 + center[1],
                    "track_source": "bytetrack" if track_id is not None else "fallback_position",
                    "bbox": bbox,
                    "center": center,
                    "ground_point": vehicle_ground_point(bbox),
                    "confidence": confidence,
                    "label": label,
                })

    return persons, vehicles


# ============================================================
# EVIDENCE SAVING
# ============================================================

def save_evidence(category, frame):
    """Saves the current frame to evidence/<category>/ with a
    microsecond-precision timestamped filename (collision-safe even
    under rapid consecutive events). The frontend's evidence
    timestamp parser supports this exact format (see
    templates/index.html / dashboard.py context helpers)."""

    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    cat_dir = os.path.join(EVIDENCE_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)
    full_path = os.path.join(cat_dir, filename)
    rel_path = f"{EVIDENCE_DIR}/{category}/{filename}".replace("\\", "/")

    try:
        cv2.imwrite(full_path, frame)
        return rel_path
    except Exception as e:
        print(f"[WARN] failed to save evidence to {full_path}: {e}")
        return None


# ============================================================
# VERIFIED-EVENT CREATION (the ONLY path that creates a user alert)
# ============================================================
# YOLO detection alone never reaches this function. Only a state
# machine transition into "verified" (littering) or a per-track zone
# counter crossing VERIFY_FRAMES for the first time in a continuous
# stay (trespassing / no-parking) calls this.

def create_verified_event(violation, category, confidence, risk, frame, bbox,
                           persistence, association=None, evidence_dir=None):

    global last_alert, active_event

    evidence_quality = compute_evidence_quality(
        frame, bbox, confidence, association if association is not None else 1.0
    )

    evidence_path = save_evidence(evidence_dir, frame)

    extra_parts = []
    if association is not None:
        extra_parts.append(f"association={association:.2f}")
    extra_parts.append(f"persistence={persistence}")
    extra = " | ".join(extra_parts)

    log_alert(violation, confidence, risk, evidence_path or "unavailable", extra=extra)

    event_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "violation": violation,
        "category": category,
        "confidence": round(float(confidence), 3),
        "risk": int(risk),
        "persistence": int(persistence),
        "evidence_quality": evidence_quality,
        "evidence": evidence_path,
    }

    if association is not None:
        event_payload["association"] = round(float(association), 3)

    saved_event = save_history(event_payload)

    last_alert = f"{violation} VERIFIED"
    active_event = saved_event

    return saved_event


# ============================================================
# PER-FRAME PROCESSING PIPELINE
# ============================================================
# Factored out of generate_frames() so it can also be exercised
# directly in tests (see test_pipeline.py) by feeding synthetic
# frames/detections without needing a live camera or real model
# weights.

def process_frame(frame):
    """Runs detection + verification for one frame, draws overlays,
    and creates verified events as appropriate. Returns the annotated
    frame. Safe to call even if models failed to load (skips
    detection, still draws zones so the operator can see the feed).
    The camera/source layer may provide a read-only NumPy view, while
    the overlay pipeline performs in-place OpenCV operations."""

    global last_garbage_save, last_trespass_save, last_nopark_save

    # OpenCV drawing and compositing functions below write into `frame`.
    # Normalize the source boundary once so webcam, file, and stream
    # frames all have the same writable behavior.
    frame = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    now = time.time()

    # -------------------- DRAW ZONES --------------------
    # Semi-transparent fill + outline so the two restricted areas are
    # immediately visually distinct from each other on the live feed,
    # not just by an outline color that's easy to miss at a glance.

    zone_overlay = frame.copy()
    cv2.fillPoly(zone_overlay, [TRESPASS_ZONE], (0, 0, 255))
    cv2.fillPoly(zone_overlay, [NO_PARKING_ZONE], (255, 0, 0))
    cv2.addWeighted(zone_overlay, 0.15, frame, 0.85, 0, frame)

    cv2.polylines(frame, [TRESPASS_ZONE], True, (0, 0, 255), 2)
    cv2.putText(frame, "TRESPASS RESTRICTED AREA",
                (int(TRESPASS_ZONE[:, 0].min()), max(20, int(TRESPASS_ZONE[:, 1].min()) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.polylines(frame, [NO_PARKING_ZONE], True, (255, 0, 0), 2)
    cv2.putText(frame, "NO PARKING AREA",
                (int(NO_PARKING_ZONE[:, 0].min()), max(20, int(NO_PARKING_ZONE[:, 1].min()) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if not models_loaded:
        cv2.putText(frame, "DETECTION UNAVAILABLE (model load failed)",
                    (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    # -------------------- DETECTION --------------------

    garbage_detections = run_garbage_detection(frame)
    persons, vehicles = run_person_vehicle_detection(frame)

    # -------------------- DRAW PERSON / VEHICLE BOXES --------------------

    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        gx, gy = p["ground_point"]
        cv2.circle(frame, (gx, gy), 4, (0, 255, 0), -1)

    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        gx, gy = v["ground_point"]
        cv2.circle(frame, (gx, gy), 4, (255, 255, 0), -1)

    # -------------------- LITTERING PIPELINE --------------------

    for g in garbage_detections:
        x1, y1, x2, y2 = g["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Always-visible label: what was detected and how confident
        # the model is, independent of verification stage — required
        # so raw garbage detections are clearly shown as detections,
        # not as violations.
        cv2.putText(frame, f"GARBAGE {g['confidence']:.2f}", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        state = process_garbage_relation(
            g["id"], g["center"], g["confidence"], persons, frame_width
        )

        stage_label = garbage_track_stage(state)
        cv2.putText(frame, stage_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        if state.get("littering_verified_this_frame") and (now - last_garbage_save) >= EVIDENCE_COOLDOWN:
            avg_confidence = (
                sum(state["confidence_history"]) / len(state["confidence_history"])
                if state["confidence_history"] else g["confidence"]
            )
            association_score = min(state["association_frames"] / 20.0, 1.0)
            risk = calculate_risk(avg_confidence, state["stationary_frames"], association_score)

            create_verified_event(
                violation="LITTERING",
                category="garbage",
                confidence=avg_confidence,
                risk=risk,
                frame=frame,
                bbox=g["bbox"],
                persistence=state["stationary_frames"],
                association=association_score,
                evidence_dir="garbage",
            )

            last_garbage_save = now
            state["last_event_time"] = now

            cv2.putText(frame, "LITTERING VERIFIED", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    # -------------------- TRESPASSING PIPELINE --------------------
    # Both tracked people and vehicles use their ground-contact point
    # for the restricted-zone test. This keeps trespassing independent
    # from the no-parking stationary requirement.

    seen_trespass_ids = set()

    for trespass_object in persons + vehicles:
        seen_trespass_ids.add(trespass_object["id"])
        inside = point_in_zone(trespass_object["ground_point"], TRESPASS_ZONE)

        verified_now, entry = update_zone_track(
            trespass_tracks,
            trespass_object["id"],
            inside,
            trespass_object["confidence"],
            None,
        )

        if inside:
            cv2.putText(frame, f"IN ZONE ({entry['frames_inside']})",
                        (trespass_object["bbox"][0], trespass_object["bbox"][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if verified_now and (now - last_trespass_save) >= EVIDENCE_COOLDOWN:
            avg_confidence = (
                sum(entry["confidence_history"]) / len(entry["confidence_history"])
                if entry["confidence_history"] else trespass_object["confidence"]
            )
            risk = calculate_risk(avg_confidence, entry["frames_inside"], context_score=1.0)

            create_verified_event(
                violation="TRESPASSING",
                category="trespassing",
                confidence=avg_confidence,
                risk=risk,
                frame=frame,
                bbox=trespass_object["bbox"],
                persistence=entry["frames_inside"],
                association=None,
                evidence_dir="trespassing",
            )

            last_trespass_save = now
            entry["last_event_time"] = now

            cv2.putText(frame, "TRESPASSING VERIFIED", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    decay_missing_tracks(trespass_tracks, seen_trespass_ids)

    # -------------------- NO-PARKING PIPELINE --------------------

    seen_vehicle_ids = set()

    for v in vehicles:
        seen_vehicle_ids.add(v["id"])
        inside = point_in_zone(v["ground_point"], NO_PARKING_ZONE)

        verified_now, entry = update_zone_track(
            nopark_tracks,
            v["id"],
            inside,
            v["confidence"],
            None,
            reference_point=v["ground_point"],
            require_stationary=True,
        )
        log_nopark_debug(v, inside, entry, verified_now)

        if inside:
            cv2.putText(
                frame,
                f"IN ZONE ({entry['frames_inside']}) "
                f"STILL ({entry['stationary_frames']})",
                (v["bbox"][0], v["bbox"][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
            )

        # `verified` is per ByteTrack ID, so a global evidence cooldown
        # must not suppress a different vehicle's first verified event.
        if verified_now:
            avg_confidence = (
                sum(entry["confidence_history"]) / len(entry["confidence_history"])
                if entry["confidence_history"] else v["confidence"]
            )
            risk = calculate_risk(avg_confidence, entry["frames_inside"], context_score=1.0)

            create_verified_event(
                violation="NO PARKING",
                category="noparking",
                confidence=avg_confidence,
                risk=risk,
                frame=frame,
                bbox=v["bbox"],
                persistence=entry["frames_inside"],
                association=None,
                evidence_dir="noparking",
            )

            last_nopark_save = now
            entry["last_event_time"] = now

            cv2.putText(frame, "NO PARKING VERIFIED", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)

    decay_missing_tracks(nopark_tracks, seen_vehicle_ids)

    return frame


# ============================================================
# CAMERA CONTROL
# ============================================================

@app.route("/start")
def start_monitoring():
    """Start monitoring from the selected video source.

    Query parameters (all optional):
      source_type  : "webcam" | "file" | "stream"  (default: "webcam")
      source_url   : file path or stream URL        (default: "" → webcam 0)
      camera_name  : display label                  (default: auto)
      fps_cap      : int, frames/sec cap for streams (default: 10)
    """
    global camera_running, cap, video_source_obj, _latest_camera_frame
    global current_source_type, current_source_label, current_source_url

    src_type  = request.args.get("source_type",  SOURCE_WEBCAM).strip()
    src_url   = request.args.get("source_url",   "").strip()
    cam_name  = request.args.get("camera_name",  "").strip()
    fps_cap   = int(request.args.get("fps_cap",  10))

    # Validate source_type
    if src_type not in (SOURCE_WEBCAM, SOURCE_FILE, SOURCE_STREAM):
        src_type = SOURCE_WEBCAM

    # Build the VideoSource (or fall back to bare cv2.VideoCapture)
    with _camera_lock:
        if camera_running:
            return redirect("/")

        with _live_frame_lock:
            _latest_camera_frame = None

        if _VIDEO_SOURCE_AVAILABLE:
            # Determine the URI to pass to make_video_source
            if src_type == SOURCE_WEBCAM:
                uri = 0
                label = cam_name or "Webcam [0]"
            elif src_type == SOURCE_FILE:
                uri = src_url or ""
                label = cam_name or (os.path.basename(src_url) if src_url else "Video File")
            else:  # stream
                uri = src_url or ""
                label = cam_name or "Public Live Stream"

            cams_cfg = load_cameras_config()
            reconnect_interval = float(
                cams_cfg.get("default_reconnect_interval", 5))
            max_reconnects = int(
                cams_cfg.get("default_max_reconnects", 20))

            # Set FFmpeg low-latency flags for stream sources BEFORE
            # creating VideoCapture — reduces buffering delay on HLS
            if src_type == SOURCE_STREAM:
                os.environ.setdefault(
                    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                    "fflags;nobuffer|flags;low_delay|rtsp_transport;tcp"
                )

            try:
                src = make_video_source(
                    source_type=src_type,
                    source_uri=uri,
                    camera_name=label,
                    fps_cap=fps_cap if src_type == SOURCE_STREAM else 0,
                    reconnect_interval=reconnect_interval,
                    max_reconnects=max_reconnects,
                )
            except Exception as e:
                print(f"[WARN] Could not create VideoSource: {e}")
                if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
                    return jsonify(ok=False, error="Could not create VideoSource"), 500
                return redirect("/")

            opened = src.open()
            if not opened:
                print(f"[WARN] VideoSource '{label}' could not be opened.")
                src.release()
                if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
                    return jsonify(ok=False, error="Stream could not be opened"), 500
                return redirect("/")

            _reset_tracking_state()
            video_source_obj = src
            cap = src._cap   # keep cap alias for snapshot endpoint
            camera_running = True
            current_source_type  = src_type
            current_source_label = label
            current_source_url   = str(uri)
            if src_type == SOURCE_STREAM:
                _start_live_processing()

        else:
            # Fallback: original bare cv2.VideoCapture(0) if
            # video_source.py is missing
            new_cap = cv2.VideoCapture(0)
            if not new_cap.isOpened():
                print("[WARN] Camera could not be opened (device unavailable).")
                new_cap.release()
                return redirect("/")
            cap = new_cap
            camera_running = True
            current_source_type  = SOURCE_WEBCAM
            current_source_label = "Webcam [0]"
            current_source_url   = ""

        if not _VIDEO_SOURCE_AVAILABLE:
            _reset_tracking_state()

    # Return JSON for AJAX callers; redirect for legacy link navigation
    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
        return jsonify(ok=True, source_type=current_source_type, source_label=current_source_label)
    return redirect("/")


@app.route("/stop")
def stop_monitoring():

    global camera_running, cap, video_source_obj, _latest_camera_frame

    with _camera_lock:
        camera_running = False

        if video_source_obj is not None:
            if current_source_type == SOURCE_STREAM:
                _stop_live_processing()
            video_source_obj.release()
            video_source_obj = None
        elif cap is not None:
            cap.release()

        cap = None
        with _live_frame_lock:
            _latest_camera_frame = None

    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
        return jsonify(ok=True)
    return redirect("/")


# ============================================================
# EVIDENCE FILES
# ============================================================

@app.route("/evidence/<category>/<filename>")
def evidence(category, filename):
    if category not in ALLOWED_EVIDENCE_CATEGORIES:
        return "Category not found", 404
    safe_filename = os.path.basename(filename.replace("\\", "/"))
    cat_dir = os.path.join(EVIDENCE_DIR, category)
    if not os.path.isdir(cat_dir):
        return "Directory not found", 404
    return send_from_directory(cat_dir, safe_filename)


# ============================================================
# TEMPLATE FILTERS FOR SAFE EVIDENCE RESOLUTION
# ============================================================

@app.template_filter("evidence_filename")
def template_evidence_filename(evidence_path):
    if not evidence_path:
        return ""
    clean = str(evidence_path).replace("\\", "/")
    return os.path.basename(clean)


@app.template_filter("evidence_url")
def template_evidence_url(event_or_path, category=None):
    if not event_or_path:
        return ""
    if isinstance(event_or_path, dict):
        if event_or_path.get("evidence_url"):
            return event_or_path["evidence_url"]
        cat = event_or_path.get("category") or "garbage"
        raw_path = event_or_path.get("evidence", "")
    else:
        cat = category or "garbage"
        raw_path = str(event_or_path)
    clean = str(raw_path).replace("\\", "/")
    filename = os.path.basename(clean)
    if not filename:
        return ""
    return f"/evidence/{cat}/{filename}"


# ============================================================
# VIDEO FEED
# ============================================================

def generate_frames():
    """MJPEG frame generator.

    Uses VideoSource.read() when available — this handles
    automatic reconnection for stream sources internally.
    Falls back to direct cap.read() if VideoSource is not loaded.
    """
    global cap, camera_running, video_source_obj, _latest_camera_frame

    while True:

        if not camera_running:
            blank = 255 * np.ones((480, 640, 3), dtype="uint8")
            cv2.putText(blank, "CAMERA OFFLINE", (170, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 60, 60), 2)
            cv2.putText(blank, "Select a source and click Start", (130, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)
            ret, buffer = cv2.imencode(".jpg", blank)
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.3)
            continue

        # --- Read next frame ---
        success = False
        frame   = None

        if video_source_obj is not None:
            # VideoSource path: handles reconnect internally
            success, frame = video_source_obj.read()

            if not success or frame is None:
                vs_status = getattr(video_source_obj, "status", "ERROR")
                vs_detail = getattr(video_source_obj, "status_detail", "")

                if vs_status == "RECONNECTING":
                    # Show reconnecting overlay and keep the stream alive
                    blank = np.zeros((480, 640, 3), dtype="uint8")
                    cv2.putText(blank, "STREAM RECONNECTING...", (100, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                    if vs_detail:
                        cv2.putText(blank, vs_detail, (150, 260),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                    ret, buffer = cv2.imencode(".jpg", blank)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buffer.tobytes() + b"\r\n")
                    continue
                else:
                    # Permanent failure or webcam/file ended
                    print(f"[WARN] VideoSource ended "
                          f"(status={vs_status}); stopping monitoring.")
                    with _camera_lock:
                        camera_running = False
                        if current_source_type == SOURCE_STREAM:
                            _stop_live_processing()
                        video_source_obj.release()
                        video_source_obj = None
                        cap = None
                    continue

        elif cap is not None:
            # Legacy fallback path (video_source.py not loaded)
            try:
                success, frame = cap.read()
            except Exception as e:
                print(f"[WARN] camera read failed: {e}")
                success = False
                frame = None

            if not success or frame is None:
                print("[WARN] camera disconnected; stopping monitoring.")
                with _camera_lock:
                    camera_running = False
                    if cap is not None:
                        cap.release()
                        cap = None
                continue
        else:
            # No source object — just show offline
            with _camera_lock:
                camera_running = False
            continue

        # --- Process and stream frame ---
        if current_source_type == SOURCE_STREAM:
            with _live_frame_lock:
                _latest_camera_frame = frame.copy()
            _submit_live_frame(frame)
            processed_frame = _latest_live_output()
            if processed_frame is not None:
                frame = processed_frame
        else:
            with _live_frame_lock:
                _latest_camera_frame = frame.copy()
            try:
                frame = process_frame(frame)
            except Exception as e:
                print(f"[WARN] frame processing error: {e}")

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ============================================================
# DASHBOARD DATA HELPERS (shared by "/" and the /api/* routes)
# ============================================================

VIOLATION_TO_CATEGORY = {
    "LITTERING": "garbage",
    "TRESPASSING": "trespassing",
    "NO PARKING": "noparking",
}

CATEGORY_LABEL = {
    "garbage": "LITTERING",
    "trespassing": "TRESPASSING",
    "noparking": "NO PARKING",
}


def _counts_from_history():
    counts = {"garbage": 0, "trespassing": 0, "noparking": 0}
    high_risk = 0
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_count = 0

    for raw_ev in violation_history:
        ev = normalize_event_record(raw_ev)
        cat = ev.get("category")
        if cat in counts:
            counts[cat] += 1
        if ev.get("risk", 0) >= HIGH_RISK_THRESHOLD:
            high_risk += 1
        if str(ev.get("timestamp", "")).startswith(today_str):
            today_count += 1

    total = counts["garbage"] + counts["trespassing"] + counts["noparking"]
    return counts, total, high_risk, today_count


def _recent_events(limit=30):
    return [normalize_event_record(ev) for ev in reversed(violation_history[-limit:])]


def _evidence_timestamp(filename):
    """Parses the evidence filename timestamp. Supports both the new
    microsecond-precision format written by this integrated pipeline
    (YYYYMMDD_HHMMSS_ffffff) and the older seconds-only format from
    earlier dashboard versions (YYYYMMDD_HHMMSS), for evidence files
    that already existed before this integration."""

    name = os.path.splitext(os.path.basename(filename))[0]

    for fmt in ("%Y%m%d_%H%M%S_%f", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(name, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    return None


def _evidence_gallery(limit_per_category=4):
    """Builds the evidence gallery from violation_history (rich
    metadata: confidence/risk/timestamp all real, from the verified
    event itself) rather than re-scanning the evidence directories
    blind. Falls back to directory listing only if an event's
    evidence field is missing evidence (e.g. save failed) —
    directory-listed items get no confidence/risk metadata, which is
    displayed honestly as unavailable rather than guessed."""

    gallery = {"garbage": [], "trespassing": [], "noparking": []}

    for raw_ev in reversed(violation_history):
        ev = normalize_event_record(raw_ev)
        cat = ev.get("category")
        if cat not in gallery or not ev.get("evidence"):
            continue
        if len(gallery[cat]) >= limit_per_category:
            continue

        filename = ev.get("evidence_filename") or os.path.basename(str(ev["evidence"]).replace("\\", "/"))
        gallery[cat].append({
            "filename": filename,
            "url": ev.get("evidence_url") or f"/evidence/{cat}/{filename}",
            "timestamp": ev.get("timestamp"),
            "confidence": ev.get("confidence"),
            "risk": ev.get("risk"),
            "violation": ev.get("violation"),
        })

    return gallery


def _risk_distribution():
    """Counts verified events by risk severity band, computed only
    from real violation_history entries — never fabricated. Bands:
    high (risk >= HIGH_RISK_THRESHOLD), medium (40-69), low (<40)."""

    bands = {"low": 0, "medium": 0, "high": 0}

    for ev in violation_history:
        risk = ev.get("risk", 0)
        if risk >= HIGH_RISK_THRESHOLD:
            bands["high"] += 1
        elif risk >= 40:
            bands["medium"] += 1
        else:
            bands["low"] += 1

    return bands


def _system_status():
    vs_status = ""
    vs_detail = ""
    if video_source_obj is not None:
        vs_status = getattr(video_source_obj, "status", "")
        vs_detail = getattr(video_source_obj, "status_detail", "")
    return {
        "camera_running": camera_running,
        "models_loaded": models_loaded,
        "garbage_model_loaded": garbage_model is not None,
        "coco_model_loaded": coco_model is not None,
        "model_load_error": model_load_error,
        "tracking_active": camera_running and models_loaded,
        "verification_active": camera_running and models_loaded,
        "logging_active": os.path.exists("logs/alerts.txt") or os.path.isdir("logs"),
        "evidence_storage_active": os.path.isdir("evidence"),
        # Video source info for UI/templates
        "source_type": current_source_type,
        "source_label": current_source_label,
        "source_url": current_source_url,
        "stream_status": vs_status,
        "stream_status_detail": vs_detail,
    }


def _dashboard_context():
    counts, total, high_risk, today_count = _counts_from_history()
    status = _system_status()

    return {
        "camera_running": camera_running,
        "status_label": "ONLINE" if camera_running else "OFFLINE",
        "garbage": counts["garbage"],
        "trespassing": counts["trespassing"],
        "noparking": counts["noparking"],
        "total_violations": total,
        "today_violations": today_count,
        "high_risk_count": high_risk,
        "risk_distribution": _risk_distribution(),
        "last_alert": last_alert,
        "active_event": normalize_event_record(active_event) if active_event else None,
        "recent_events": _recent_events(30),
        "evidence": _evidence_gallery(4),
        "trespass_zone": TRESPASS_ZONE.tolist(),
        "no_parking_zone": NO_PARKING_ZONE.tolist(),
        **status,
    }


# ============================================================
# DASHBOARD PAGE (first paint — subsequent updates via /api/*)
# ============================================================

@app.route("/")
def home():
    return render_template("index.html", **_dashboard_context())


# ============================================================
# ZONE CALIBRATION API
# ============================================================
# Lets the operator define the real restricted floor area for their
# camera instead of using the generic default trapezoids. GET
# returns the current polygons; POST updates one or both (each needs
# >= 3 points), persists them to zones_config.json, and clears the
# affected zone's per-track state (frames_inside counters) since the
# geometry changed underneath any tracks currently mid-count — this
# does NOT touch logs/evidence/violation_history at all.

@app.route("/api/zones", methods=["GET", "POST"])
def api_zones():
    global TRESPASS_ZONE, NO_PARKING_ZONE

    if request.method == "GET":
        return jsonify({
            "trespass_zone": TRESPASS_ZONE.tolist(),
            "no_parking_zone": NO_PARKING_ZONE.tolist(),
        })

    data = request.get_json(force=True, silent=True) or {}
    errors = []
    parsed = {}

    for key in ("trespass_zone", "no_parking_zone"):
        pts = data.get(key)
        if pts is None:
            continue
        if not isinstance(pts, list) or len(pts) < 3:
            errors.append(f"{key} needs at least 3 points")
            continue
        try:
            arr = np.array([[int(round(float(p[0]))), int(round(float(p[1])))] for p in pts], dtype=np.int32)
        except Exception:
            errors.append(f"{key} contains invalid point data")
            continue
        parsed[key] = arr

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    if not parsed:
        return jsonify({"ok": False, "errors": ["no valid zone data provided"]}), 400

    if "trespass_zone" in parsed:
        TRESPASS_ZONE = parsed["trespass_zone"]
        trespass_tracks.clear()

    if "no_parking_zone" in parsed:
        NO_PARKING_ZONE = parsed["no_parking_zone"]
        nopark_tracks.clear()

    save_zones_config()

    return jsonify({
        "ok": True,
        "trespass_zone": TRESPASS_ZONE.tolist(),
        "no_parking_zone": NO_PARKING_ZONE.tolist(),
    })


@app.route("/api/snapshot")
def api_snapshot():
    """Return the most recent frame already read by the live camera feed."""

    frame = None
    deadline = time.time() + 1.5
    while camera_running and time.time() < deadline:
        with _live_frame_lock:
            if _latest_camera_frame is not None:
                frame = _latest_camera_frame.copy()
                break
        time.sleep(0.05)

    frame_available = frame is not None
    if frame is None:
        frame = 255 * np.ones((480, 640, 3), dtype="uint8")
        message = (
            "Start the camera for a live snapshot"
            if not camera_running else
            "Waiting for the camera frame..."
        )
        cv2.putText(frame, message, (40, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2)
        cv2.putText(frame, "Refresh after the live feed is ready", (40, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    ret, buffer = cv2.imencode(".jpg", frame)
    response = Response(buffer.tobytes(), mimetype="image/jpeg")
    response.headers["X-Camera-Frame"] = "available" if frame_available else "unavailable"
    return response


# ============================================================
# REAL-TIME DASHBOARD API
# ============================================================
# Polled by static/js/dashboard.js roughly every 1.5-2s. Only a
# VERIFIED event (created exclusively by create_verified_event, never
# by raw detection) ever appears here — this is what the frontend
# alert/toast system is driven from.

@app.route("/api/status")
def api_status():
    counts, total, high_risk, today_count = _counts_from_history()
    status = _system_status()

    return jsonify({
        "counts": {
            "littering": counts["garbage"],
            "trespassing": counts["trespassing"],
            "noparking": counts["noparking"],
            "total": total,
            "today": today_count,
            "high_risk": high_risk,
        },
        "risk_distribution": _risk_distribution(),
        "last_alert": last_alert,
        "active_event": normalize_event_record(active_event) if active_event else None,
        "last_event_id": active_event.get("id", 0) if active_event else 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **status,
    })


@app.route("/api/latest-event")
def api_latest_event():
    """Raw JSON of the single most recent verified event (or null).
    Used by the frontend to build toast/alert text without needing
    to parse rendered HTML."""
    return jsonify(normalize_event_record(active_event) if active_event else None)


@app.route("/api/events")
def api_events():
    """Server-rendered HTML fragments for the sections that need to
    refresh when a new verified event occurs (Active Event panel,
    Recent Violations table, Evidence gallery, System Status). Reuses
    the exact same Jinja partials as the initial page load, so there
    is a single source of truth for markup — the frontend just swaps
    innerHTML rather than re-implementing table/card rendering in
    JavaScript."""

    ctx = _dashboard_context()

    return jsonify({
        "active_event_html": render_template("_active_event.html", **ctx),
        "violations_table_html": render_template("_violations_table.html", **ctx),
        "evidence_gallery_html": render_template("_evidence_gallery.html", **ctx),
        "system_status_html": render_template("_system_status.html", **ctx),
        "counts": {
            "littering": ctx["garbage"],
            "trespassing": ctx["trespassing"],
            "noparking": ctx["noparking"],
            "total": ctx["total_violations"],
            "today": ctx["today_violations"],
            "high_risk": ctx["high_risk_count"],
        },
        "risk_distribution": ctx["risk_distribution"],
    })



# ============================================================
# VIDEO SOURCE API
# ============================================================

@app.route("/api/cameras")
def api_cameras():
    """Returns the list of public demo camera presets from
    config/cameras.json. Used by the frontend source selector
    to populate the preset dropdown."""
    cfg = load_cameras_config()
    streams = [
        {
            "id": s.get("id", ""),
            "name": s.get("name", ""),
            "url": s.get("url", ""),
            "format": s.get("format", ""),
            "notes": s.get("notes", ""),
            "fps_cap": s.get("fps_cap", 10),
            "enabled": s.get("enabled", True),
        }
        for s in cfg.get("demo_streams", [])
        if s.get("enabled", True)
    ]
    return jsonify({"streams": streams})


@app.route("/api/source")
def api_source():
    """Returns the current video source state for the UI status panel."""
    vs_info = {}
    if video_source_obj is not None:
        vs_info = video_source_obj.get_info() if hasattr(video_source_obj, "get_info") else {}
    return jsonify({
        "camera_running": camera_running,
        "source_type": current_source_type,
        "source_label": current_source_label,
        "source_url": current_source_url if current_source_type != SOURCE_WEBCAM else "",
        "stream_status": vs_info.get("status", "OFFLINE" if not camera_running else "LIVE"),
        "stream_status_detail": vs_info.get("status_detail", ""),
        "reconnect_count": vs_info.get("reconnect_count", 0),
        "width": vs_info.get("width", 640),
        "height": vs_info.get("height", 480),
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
