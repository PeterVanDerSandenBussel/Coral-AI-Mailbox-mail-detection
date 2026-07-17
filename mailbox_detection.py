import cv2
import numpy as np
import time
import math
import threading
import os
import json
import datetime
import configparser
import paho.mqtt.client as mqtt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from flask import Flask, Response
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import detect
from waitress import serve
from concurrent.futures import ThreadPoolExecutor

# --- 1. CONFIGURATION (read from config.ini) ---
_cfg = configparser.ConfigParser()
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
if not _cfg.read(_cfg_path):
    raise FileNotFoundError(f"config.ini not found at {_cfg_path}")

MODEL_PATH = _cfg.get('camera', 'model_path')
LABEL_PATH = _cfg.get('camera', 'label_path')
RTSP_URL   = _cfg.get('camera', 'rtsp_url')

# RTSP connection/read timeouts -- without these, cv2.VideoCapture.read() can
# block forever on a stalled stream, which leaves the reader thread stuck and
# leaks a duplicate RTSP connection to the camera every time the watchdog
# tries (and fails) to restart it. This eventually exhausts the camera's
# connection limit, blocking both this app and other RTSP viewers.
CAM_OPEN_TIMEOUT_MS  = _cfg.getint('camera', 'open_timeout_ms')  if _cfg.has_option('camera', 'open_timeout_ms')  else 5000
CAM_READ_TIMEOUT_MS  = _cfg.getint('camera', 'read_timeout_ms')  if _cfg.has_option('camera', 'read_timeout_ms')  else 5000
CAM_WATCHDOG_SECONDS = _cfg.getint('camera', 'watchdog_seconds') if _cfg.has_option('camera', 'watchdog_seconds') else 8
CAM_RECONNECT_DELAY  = _cfg.getfloat('camera', 'reconnect_delay') if _cfg.has_option('camera', 'reconnect_delay') else 3.0

MQTT_BROKER   = _cfg.get('mqtt', 'broker')
MQTT_PORT     = _cfg.getint('mqtt', 'port')
MQTT_USER     = _cfg.get('mqtt', 'user')
MQTT_PW       = _cfg.get('mqtt', 'pw')
MQTT_TOPIC    = _cfg.get('mqtt', 'topic')
MQTT_INTERVAL = _cfg.getfloat('mqtt', 'interval')

MAIL_ENABLED     = _cfg.getboolean('mail', 'enabled')
MAIL_SMTP_SERVER = _cfg.get('mail', 'smtp_server')
MAIL_SMTP_PORT   = _cfg.getint('mail', 'smtp_port')
MAIL_FROM        = _cfg.get('mail', 'from_addr')
MAIL_FROM_NAME   = _cfg.get('mail', 'from_name')
MAIL_TO          = _cfg.get('mail', 'to_addr')

MIN_CONFIDENCE_DAY   = _cfg.getfloat('detectie', 'min_confidence_day')
MIN_CONFIDENCE_NIGHT = _cfg.getfloat('detectie', 'min_confidence_night')
NIGHT_INTERVAL       = _cfg.getfloat('detectie', 'night_interval')
TARGET_LABELS        = [l.strip() for l in _cfg.get('detectie', 'target_labels').split(',')]

# =============================================================================
# MAIL DELIVERY DETECTION CONFIGURATION
#
# All zones are in AI coordinates (0-300), since cx_ai/cy_ai are AI coordinates.
#
# Mail carrier pattern (from logging):
#   ARRIVAL : from top-right  cx > ARRIVE_MIN_CX, cy < ARRIVE_MAX_CY
#   MAILBOX : cx ~ 78-102, cy ~ 130-188  MAILBOX_ZONE_AI
#   DEPARTURE: back to top-right       cx > DEPART_MIN_CX
#
# EXCLUDE_ZONE_AI: area top-right where passersby walk
#   AI [200,0] through [300,70] = top-right corner, passerby path
# =============================================================================

def _parse_zone(s):
    return tuple(int(x.strip()) for x in s.split(','))

MAILBOX_ZONE_AI = _parse_zone(_cfg.get('post_detectie', 'mailbox_zone'))
EXCLUDE_ZONE_AI = _parse_zone(_cfg.get('post_detectie', 'exclude_zone'))

ARRIVE_MIN_CX        = _cfg.getfloat('post_detectie', 'arrive_min_cx')
ARRIVE_MAX_CY        = _cfg.getfloat('post_detectie', 'arrive_max_cy')
DEPART_MIN_CX        = _cfg.getfloat('post_detectie', 'depart_min_cx')
DEPART_MAX_CY        = _cfg.getfloat('post_detectie', 'depart_max_cy')
MAILBOX_DWELL_MIN    = _cfg.getfloat('post_detectie', 'mailbox_dwell_min')
MAILBOX_DWELL_MAX    = _cfg.getfloat('post_detectie', 'mailbox_dwell_max')
APPROACH_MIN_SECONDS = _cfg.getfloat('post_detectie', 'approach_min_seconds')
POST_COOLDOWN_SECONDS= _cfg.getfloat('post_detectie', 'post_cooldown')
# Require actual leftward/rightward movement (not just position) before
# confirming arrival at / departure from the mailbox. Set to false in
# config.ini if this turns out too strict for your camera angle.
REQUIRE_DIRECTION_CHECK = _cfg.getboolean('post_detectie', 'require_direction_check') \
    if _cfg.has_option('post_detectie', 'require_direction_check') else True

TRACKER_MAX_DISAPPEARED  = _cfg.getint('tracker', 'max_disappeared')
TRACKER_RESET_SECONDS    = _cfg.getint('tracker', 'reset_seconds')
TRACKER_MAX_TRACE_POINTS = _cfg.getint('tracker', 'max_trace_points')

# debug overlay
DRAW_DEBUG        = _cfg.getboolean('debug', 'draw_debug')
DRAW_TRACES_DEBUG = _cfg.getboolean('debug', 'draw_traces_debug')

# Camera load: maximum frame rate for the reader thread and detection loop.
# Hikvision typically delivers 25fps; 8fps is more than enough for walking detection.
# Also adjustable in config.ini if desired: [camera] reader_fps / detect_fps
READER_FPS  = _cfg.getfloat('camera', 'reader_fps')   if _cfg.has_option('camera', 'reader_fps')  else 15.0
DETECT_FPS  = _cfg.getfloat('camera', 'detect_fps')   if _cfg.has_option('camera', 'detect_fps')  else  8.0
_READER_INTERVAL = 1.0 / READER_FPS   # seconds between reads
_DETECT_INTERVAL = 1.0 / DETECT_FPS   # seconds between detections

output_frame = None
lock = threading.Lock()
is_night = False

app = Flask(__name__)

_async_executor = ThreadPoolExecutor(max_workers=2)

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id="")
client.username_pw_set(MQTT_USER, MQTT_PW)
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()


# =============================================================================
# EMAIL FUNCTION
# =============================================================================
def send_post_email(jpeg_bytes):
    """Sends an email with a snapshot when mail delivery is detected. Runs via the executor."""
    if not MAIL_ENABLED:
        return
    def _send(data):
        try:
            ts = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            msg = MIMEMultipart()
            msg['Subject'] = f"Mail delivered! 📬 {ts}"
            msg['From']    = f"{MAIL_FROM_NAME} <{MAIL_FROM}>"
            msg['To']      = MAIL_TO
            msg.attach(MIMEText(f"The mail carrier was detected at {ts}.", 'plain'))
            msg.attach(MIMEImage(data, name="post_snapshot.jpg"))
            with smtplib.SMTP(MAIL_SMTP_SERVER, MAIL_SMTP_PORT, timeout=10) as smtp:
                smtp.sendmail(MAIL_FROM, MAIL_TO, msg.as_string())
            print(f"[MAIL] Sent to {MAIL_TO} at {ts}")
        except Exception as e:
            print(f"[MAIL] Error: {e}")
    _async_executor.submit(_send, jpeg_bytes)


# =============================================================================
# HELPER FUNCTIONS: zone check in AI coordinates
# =============================================================================
def point_in_zone_ai(cx_ai, cy_ai, zone_ai):
    """Checks whether an AI centroid falls within an AI zone."""
    return zone_ai[0] <= cx_ai <= zone_ai[2] and zone_ai[1] <= cy_ai <= zone_ai[3]

def ai_to_px(ai_x, ai_y, frame_w, frame_h):
    """Converts AI coordinates (300x300) to pixel coordinates for drawing."""
    size = min(frame_h, frame_w)
    start_x = (frame_w - size) // 2
    start_y = (frame_h - size) // 2
    px_x = int(ai_x * size / 300.0) + start_x
    px_y = int(ai_y * size / 300.0) + start_y
    return px_x, px_y

def ai_zone_to_px(zone_ai, frame_w, frame_h):
    """Converts an AI zone to a pixel rectangle for drawing."""
    x1, y1 = ai_to_px(zone_ai[0], zone_ai[1], frame_w, frame_h)
    x2, y2 = ai_to_px(zone_ai[2], zone_ai[3], frame_w, frame_h)
    return x1, y1, x2, y2


# =============================================================================
# STATE MACHINE PER PERSON
#
# Three hard requirements for POST_CONFIRMED:
#   1. ARRIVAL  - track starts on the right (cx > ARRIVE_MIN_CX), not in the exclude zone,
#                 and moves toward the bottom-left, toward the mailbox
#   2. DELIVERY - person in MAILBOX_ZONE_AI for between MAILBOX_DWELL_MIN and MAILBOX_DWELL_MAX sec
#   3. DEPARTURE- person leaves the mailbox and moves back to the right (cx > DEPART_MIN_CX)
#
# False positives are prevented by:
#   - Exclude zone top-right: passersby who are not heading to the mailbox
#   - Direction requirement on arrival: cx must decrease on the way to the mailbox
#   - Direction requirement on departure: cx must increase after the mailbox
#   - Max dwell: standing still too long = not the mail carrier
# =============================================================================
POST_STATE_IDLE      = "IDLE"
POST_STATE_APPROACH  = "APPROACHING"
POST_STATE_MAILBOX   = "AT_MAILBOX"
POST_STATE_DEPARTING = "DEPARTING"
POST_STATE_CONFIRMED = "POST_CONFIRMED"


class PostTracker:
    """Tracks the mail delivery status per person ID with direction detection."""
    def __init__(self):
        self.states        = {}   # obj_id -> state
        self.approach_entry= {}   # obj_id -> timestamp of approach start
        self.mailbox_entry = {}   # obj_id -> timestamp of mailbox arrival
        self.confirmed_time= {}   # obj_id -> timestamp of confirmation
        self.first_cx      = {}   # obj_id -> first cx (for arrival direction check)
        self.cx_history    = {}   # obj_id -> list of recent cx values
        self.arrived_from_right = set()  # IDs that demonstrably came from the right

    def get_state(self, obj_id):
        return self.states.get(obj_id, POST_STATE_IDLE)

    def _in_exclude(self, cx_ai, cy_ai):
        return point_in_zone_ai(cx_ai, cy_ai, EXCLUDE_ZONE_AI)

    def _moving_left(self, obj_id, cx_ai):
        """Is the person moving left (cx decreasing)?"""
        hist = self.cx_history.get(obj_id, [])
        if len(hist) < 3:
            return False
        avg_prev = sum(hist[-3:]) / 3
        return cx_ai < avg_prev - 2

    def _moving_right(self, obj_id, cx_ai):
        """Is the person moving right (cx increasing)?"""
        hist = self.cx_history.get(obj_id, [])
        if len(hist) < 3:
            return False
        avg_prev = sum(hist[-3:]) / 3
        return cx_ai > avg_prev + 2

    def update(self, obj_id, cx_ai, cy_ai, frame_w, frame_h):
        now = time.time()
        current_state = self.states.get(obj_id, POST_STATE_IDLE)

        in_mailbox = point_in_zone_ai(cx_ai, cy_ai, MAILBOX_ZONE_AI)
        in_exclude = self._in_exclude(cx_ai, cy_ai)

        # Track cx history for direction detection
        if obj_id not in self.cx_history:
            self.cx_history[obj_id] = []
        self.cx_history[obj_id].append(cx_ai)
        if len(self.cx_history[obj_id]) > 10:
            self.cx_history[obj_id].pop(0)

        # Record first cx
        if obj_id not in self.first_cx:
            self.first_cx[obj_id] = cx_ai

        print(f"[ZONE] ID={obj_id} cx={cx_ai:.0f} cy={cy_ai:.0f} "
              f"mailbox={in_mailbox} excl={in_exclude} state={current_state}")

        # --- Cooldown after confirmation ---
        if current_state == POST_STATE_CONFIRMED:
            if now - self.confirmed_time.get(obj_id, 0) > POST_COOLDOWN_SECONDS:
                self._reset(obj_id)
            return self.states.get(obj_id, POST_STATE_IDLE), True

        # =========================================================
        # PHASE 3 first: DEPARTING - check whether person leaves to the right
        # =========================================================
        if current_state == POST_STATE_DEPARTING:
            at_depart_pos = not in_mailbox and cx_ai > DEPART_MIN_CX and cy_ai < DEPART_MAX_CY
            if at_depart_pos and (not REQUIRE_DIRECTION_CHECK or self._moving_right(obj_id, cx_ai)):
                self._confirm(obj_id, now)
                return POST_STATE_CONFIRMED, True
            return POST_STATE_DEPARTING, False

        # =========================================================
        # PHASE 2: AT the mailbox
        # =========================================================
        elif current_state == POST_STATE_MAILBOX:
            if in_mailbox:
                dwell = now - self.mailbox_entry.get(obj_id, now)
                if dwell > MAILBOX_DWELL_MAX:
                    print(f"[POST] ID={obj_id} -> RESET (dwell too long: {dwell:.1f}s)")
                    self._reset(obj_id)
                elif dwell >= MAILBOX_DWELL_MIN:
                    self.states[obj_id] = POST_STATE_DEPARTING
                    print(f"[POST] ID={obj_id} -> DEPARTING (dwell={dwell:.1f}s)")
            else:
                dwell = now - self.mailbox_entry.get(obj_id, now)
                if dwell < MAILBOX_DWELL_MIN:
                    print(f"[POST] ID={obj_id} -> back to APPROACH (dwell too short: {dwell:.1f}s)")
                    self.states[obj_id] = POST_STATE_APPROACH
                else:
                    self.states[obj_id] = POST_STATE_DEPARTING
                    print(f"[POST] ID={obj_id} -> DEPARTING (left mailbox after {dwell:.1f}s)")

        # =========================================================
        # PHASE 1b: APPROACHING - on the way to the mailbox
        # =========================================================
        elif current_state == POST_STATE_APPROACH:
            if in_exclude:
                print(f"[POST] ID={obj_id} -> RESET (in exclude zone)")
                self._reset(obj_id)
            elif in_mailbox:
                approach_time = now - self.approach_entry.get(obj_id, now)
                moving_toward = not REQUIRE_DIRECTION_CHECK or self._moving_left(obj_id, cx_ai)
                if approach_time >= APPROACH_MIN_SECONDS and obj_id in self.arrived_from_right and moving_toward:
                    self.mailbox_entry[obj_id] = now
                    self.states[obj_id] = POST_STATE_MAILBOX
                    print(f"[POST] ID={obj_id} -> AT_MAILBOX (approach={approach_time:.1f}s)")
                else:
                    if approach_time < APPROACH_MIN_SECONDS:
                        reason = "approach too short"
                    elif obj_id not in self.arrived_from_right:
                        reason = "did not arrive from the right"
                    else:
                        reason = "not moving toward mailbox"
                    print(f"[POST] ID={obj_id} -> RESET ({reason})")
                    self._reset(obj_id)

        # =========================================================
        # PHASE 1a: IDLE - waiting for arrival from the right
        # =========================================================
        elif current_state == POST_STATE_IDLE:
            if in_exclude:
                pass
            elif cx_ai > ARRIVE_MIN_CX and cy_ai < ARRIVE_MAX_CY and not in_mailbox:
                self.arrived_from_right.add(obj_id)
                self.approach_entry[obj_id] = now
                self.states[obj_id] = POST_STATE_APPROACH
                print(f"[POST] ID={obj_id} -> APPROACHING (cx={cx_ai:.0f} cy={cy_ai:.0f})")
            elif in_mailbox:
                pass

        return self.states.get(obj_id, POST_STATE_IDLE), False

    def _confirm(self, obj_id, now):
        self.states[obj_id] = POST_STATE_CONFIRMED
        self.confirmed_time[obj_id] = now
        print(f"[POST] ID={obj_id} DELIVERY CONFIRMED at {datetime.datetime.now().strftime('%H:%M:%S')}")

    def _reset(self, obj_id):
        for d in (self.states, self.approach_entry, self.mailbox_entry,
                  self.confirmed_time, self.first_cx, self.cx_history):
            d.pop(obj_id, None)
        self.arrived_from_right.discard(obj_id)

    def remove(self, obj_id):
        self._reset(obj_id)


# =============================================================================
# CENTROID TRACKER
# =============================================================================
class CentroidTracker:
    def __init__(self, max_disappeared=TRACKER_MAX_DISAPPEARED,
                 reset_after_seconds=TRACKER_RESET_SECONDS,
                 max_trace_points=TRACKER_MAX_TRACE_POINTS):
        self.next_id = 0
        self.objects = {}
        self.labels = {}
        self.disappeared = {}
        self.reported_ids = set()
        self.traces = {}
        self.max_disappeared = max_disappeared
        self.max_trace_points = max_trace_points
        self.last_seen_time = time.time()
        self.reset_after_seconds = reset_after_seconds
        self._post_tracker = PostTracker()
        self.pending_post_ids = set()
        self.mail_sent_ids = set()

    @property
    def post_tracker(self):
        return self._post_tracker

    def _expire_if_stale(self, obj_id):
        """Bumps the disappeared-counter for an object that wasn't matched this
        frame; deregisters it once it exceeds max_disappeared. Also confirms a
        pending delivery if the object vanished top-right while DEPARTING."""
        if obj_id not in self.disappeared:
            return
        self.disappeared[obj_id] += 1
        if self.disappeared[obj_id] > self.max_disappeared:
            state = self._post_tracker.get_state(obj_id)
            if state == POST_STATE_DEPARTING:
                last_pos = self.objects.get(obj_id, (0, 0))
                last_cx, last_cy = last_pos[0], last_pos[1]
                if last_cx > DEPART_MIN_CX and last_cy < DEPART_MAX_CY:
                    print(f"[POST] ID={obj_id} disappeared top-right during DEPARTING -> DELIVERY CONFIRMED")
                    self._post_tracker._confirm(obj_id, time.time())
                    self.pending_post_ids.add(obj_id)
                else:
                    print(f"[POST] ID={obj_id} disappeared but position cx={last_cx:.0f} cy={last_cy:.0f} doesn't match, no delivery")
            self._post_tracker.remove(obj_id)
            self.deregister(obj_id)

    def update(self, rects, labels_list):
        current_time = time.time()
        if (current_time - self.last_seen_time) > self.reset_after_seconds:
            self.__init__(self.max_disappeared, self.reset_after_seconds, self.max_trace_points)

        if not rects:
            for obj_id in list(self.objects.keys()):
                self._expire_if_stale(obj_id)
            return self.objects

        input_centroids = []
        for (xmin, ymin, xmax, ymax) in rects:
            input_centroids.append((int((xmin + xmax) / 2.0), int((ymin + ymax) / 2.0)))

        if not self.objects:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], labels_list[i])
        else:
            object_ids = list(self.objects.keys())
            registered_this_frame = set()
            matched_inputs = set()

            # Build all valid (distance, input_idx, obj_id) candidates -- same
            # label, within the match radius -- then assign globally
            # nearest-first. This prevents an input processed earlier from
            # grabbing a "good enough" ID that actually belongs to another,
            # closer input later in the list (which can swap IDs between two
            # nearby same-label objects).
            candidates = []
            for i, i_centroid in enumerate(input_centroids):
                i_label = labels_list[i]
                for oid in object_ids:
                    if self.labels[oid] == i_label:
                        d = math.hypot(i_centroid[0] - self.objects[oid][0],
                                       i_centroid[1] - self.objects[oid][1])
                        if d < 60:
                            candidates.append((d, i, oid))
            candidates.sort(key=lambda c: c[0])

            for d, i, oid in candidates:
                if i in matched_inputs or oid in registered_this_frame:
                    continue
                i_centroid = input_centroids[i]
                self.objects[oid] = i_centroid
                self.disappeared[oid] = 0
                registered_this_frame.add(oid)
                matched_inputs.add(i)

                if oid not in self.traces:
                    self.traces[oid] = []
                self.traces[oid].append(i_centroid)
                if len(self.traces[oid]) > self.max_trace_points:
                    self.traces[oid].pop(0)

            for i, i_centroid in enumerate(input_centroids):
                if i not in matched_inputs:
                    self.register(i_centroid, labels_list[i])

            # Objects that existed before this frame but weren't matched are
            # one frame closer to expiring, regardless of whether other
            # objects were detected this frame.
            for oid in object_ids:
                if oid not in registered_this_frame:
                    self._expire_if_stale(oid)

        self.last_seen_time = current_time
        return self.objects

    def register(self, centroid, label):
        self.objects[self.next_id] = centroid
        self.labels[self.next_id] = label
        self.disappeared[self.next_id] = 0
        self.traces[self.next_id] = [centroid]
        self.next_id += 1

    def deregister(self, obj_id):
        self.objects.pop(obj_id, None)
        self.labels.pop(obj_id, None)
        self.disappeared.pop(obj_id, None)
        self.traces.pop(obj_id, None)


# =============================================================================
# STATE COLORS for overlay
# =============================================================================
STATE_COLORS = {
    POST_STATE_IDLE:      (180, 180, 180),  # Gray
    POST_STATE_APPROACH:  (0, 200, 255),    # Yellow-orange
    POST_STATE_MAILBOX:   (0, 165, 255),    # Orange
    POST_STATE_DEPARTING: (0, 100, 255),    # Orange-red
    POST_STATE_CONFIRMED: (0, 0, 255),      # Red (delivery confirmed!)
}


# =============================================================================
# RTSP CAMERA READER - separate thread with watchdog
# =============================================================================
class RTSPReader:
    WATCHDOG_SECONDS = CAM_WATCHDOG_SECONDS  # Reconnect after N seconds with no new frame

    def __init__(self, url):
        self.url = url
        self._frame = None
        self._last_frame_time = 0
        self._lock = threading.Lock()
        self._stop_reader = False
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        self._wd_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._wd_thread.start()

    def _make_cap(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        c = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Without these, cap.read() can block forever on a stalled stream
        # instead of failing fast so the reader loop can reconnect.
        open_prop = getattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC', 53)
        read_prop = getattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC', 54)
        c.set(open_prop, CAM_OPEN_TIMEOUT_MS)
        c.set(read_prop, CAM_READ_TIMEOUT_MS)
        return c

    def _reader_loop(self):
        cap = self._make_cap()
        while not self._stop_reader:
            t0 = time.time()
            ret, frame = cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._last_frame_time = time.time()
                # Throttle: wait out the rest of the frame interval so we
                # don't poll the camera faster than READER_FPS (default 15 fps).
                elapsed = time.time() - t0
                wait = _READER_INTERVAL - elapsed
                if wait > 0:
                    time.sleep(wait)
            else:
                print("[CAM] cap.read() failed -- reconnecting...")
                cap.release()
                time.sleep(CAM_RECONNECT_DELAY)
                if not self._stop_reader:
                    cap = self._make_cap()
        cap.release()
        print("[CAM] Reader thread stopped.")

    def _watchdog_loop(self):
        """Detects a stalled stream."""
        while True:
            time.sleep(3)
            if self._last_frame_time == 0:
                continue
            age = time.time() - self._last_frame_time
            if age > self.WATCHDOG_SECONDS:
                print(f"[CAM] Watchdog: no frame for {age:.0f}s -- restarting stream")
                self._stop_reader = True
                # Reader may currently be blocked inside cap.read(); give it enough
                # time to hit the read timeout + reconnect-sleep before giving up.
                join_timeout = (CAM_READ_TIMEOUT_MS / 1000.0) + CAM_RECONNECT_DELAY + 2
                self._thread.join(timeout=join_timeout)
                if self._thread.is_alive():
                    print(f"[CAM] Warning: old reader thread still active after {join_timeout:.0f}s timeout")
                # Only now start a new thread (no more duplicate RTSP connection)
                self._stop_reader = False
                self._last_frame_time = time.time()  # prevent immediate re-trigger
                self._thread = threading.Thread(target=self._reader_loop, daemon=True)
                self._thread.start()
                print("[CAM] Reader thread restarted.")

    def read(self):
        """Returns (True, frame) or (False, None) -- never blocking."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()


# =============================================================================
# MAIN DETECTION LOOP
# =============================================================================
def detect_objects():
    global output_frame, is_night
    last_mqtt_time = 0
    last_grey_time = 0

    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    labels = read_label_file(LABEL_PATH)

    camera = RTSPReader(RTSP_URL)
    print("[CAM] Waiting for first frame...")
    while True:
        ok, _ = camera.read()
        if ok:
            break
        time.sleep(0.2)
    print("[CAM] Stream active.")

    tracker = CentroidTracker()

    last_detect_time = 0.0

    while True:
        try:
            # Throttle the detection loop to DETECT_FPS (default 8 fps).
            # Prevents the Coral + frame resizing from running unnecessarily fast.
            now_t = time.time()
            sleep_t = _DETECT_INTERVAL - (now_t - last_detect_time)
            if sleep_t > 0:
                time.sleep(sleep_t)
            last_detect_time = time.time()

            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            h, w, _ = frame.shape
            grey_curr_t = time.time()
            if grey_curr_t - last_grey_time > NIGHT_INTERVAL:
                last_grey_time = grey_curr_t
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                is_night = np.mean(gray) < 70
            conf = MIN_CONFIDENCE_NIGHT if is_night else MIN_CONFIDENCE_DAY

            size = min(h, w)
            start_x, start_y = (w - size) // 2, (h - size) // 2
            crop = frame[start_y:start_y + size, start_x:start_x + size]
            input_frame = cv2.resize(crop, (300, 300))

            common.set_input(interpreter, input_frame)
            interpreter.invoke()
            objs = detect.get_objects(interpreter, conf)

            rects, frame_labels, valid_objs = [], [], []
            for obj in objs:
                label_name = labels.get(obj.id, obj.id)
                if label_name in TARGET_LABELS:
                    rects.append([obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax])
                    frame_labels.append(label_name)
                    valid_objs.append(obj)

            tracked_objects = tracker.update(rects, frame_labels)
            trace_frame = frame.copy()

            if DRAW_DEBUG:
                overlay = trace_frame.copy()
                mx1, my1, mx2, my2 = ai_zone_to_px(MAILBOX_ZONE_AI, w, h)
                cv2.rectangle(overlay, (mx1, my1), (mx2, my2), (0, 0, 200), -1)
                ex1, ey1, ex2, ey2 = ai_zone_to_px(EXCLUDE_ZONE_AI, w, h)
                cv2.rectangle(overlay, (ex1, ey1), (ex2, ey2), (150, 0, 150), -1)
                arx, ary = ai_to_px(ARRIVE_MIN_CX, ARRIVE_MAX_CY, w, h)
                cv2.addWeighted(overlay, 0.18, trace_frame, 0.82, 0, trace_frame)
                cv2.line(trace_frame, (arx, 0), (arx, h), (0, 180, 180), 1)
                cv2.line(trace_frame, (0, ary), (w, ary), (0, 180, 180), 1)
                cv2.putText(trace_frame, "MAILBOX", (mx1 + 2, my1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 255), 1)
                cv2.putText(trace_frame, "EXCLUDE", (ex1 + 2, ey1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 0, 180), 1)

            if DRAW_TRACES_DEBUG:
                for oid, points in tracker.traces.items():
                    if len(points) < 2:
                        continue
                    full_scale_points = [ai_to_px(cx, cy, w, h) for (cx, cy) in points]
                    pts_arr = np.array(full_scale_points, np.int32).reshape((-1, 1, 2))
                    post_state = tracker.post_tracker.get_state(oid)
                    trace_color = STATE_COLORS.get(post_state, (255, 100, 0))
                    cv2.polylines(trace_frame, [pts_arr], False, trace_color, 2)

            # --- Process detected objects ---
            found_objects = []
            for i, obj in enumerate(valid_objs):
                label_name = frame_labels[i]
                cx_ai = (obj.bbox.xmin + obj.bbox.xmax) / 2.0
                cy_ai = (obj.bbox.ymin + obj.bbox.ymax) / 2.0

                this_id = None
                for oid, centroid in tracked_objects.items():
                    if (tracker.labels[oid] == label_name and
                            math.hypot(cx_ai - centroid[0], cy_ai - centroid[1]) < 40):
                        this_id = oid
                        break
                if this_id is None:
                    continue

                if point_in_zone_ai(cx_ai, cy_ai, EXCLUDE_ZONE_AI):
                    continue

                post_state = POST_STATE_IDLE
                is_post_confirmed = False
                if label_name == 'persoon':
                    post_state, is_post_confirmed = tracker.post_tracker.update(
                        this_id, cx_ai, cy_ai, w, h
                    )

                is_new = (this_id not in tracker.reported_ids)

                if DRAW_DEBUG:
                    bbox_s = obj.bbox.scale(size / 300.0, size / 300.0)
                    x1 = int(bbox_s.xmin) + start_x
                    y1 = int(bbox_s.ymin) + start_y
                    x2 = int(bbox_s.xmax) + start_x
                    y2 = int(bbox_s.ymax) + start_y
                    color = STATE_COLORS.get(post_state, (0, 255, 0))
                    cv2.rectangle(trace_frame, (x1, y1), (x2, y2), color, 2)
                    label_txt = f"ID:{this_id} {label_name} [{post_state[:3]}]"
                    cv2.putText(trace_frame, label_txt, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    if is_post_confirmed:
                        cv2.putText(trace_frame, "MAIL DELIVERED!", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                found_objects.append({
                    "id": this_id,
                    "label": label_name,
                    "score": int(obj.score * 100),
                    "is_new_detection": is_new,
                    "is_post": is_post_confirmed,
                    "post_state": post_state,
                    "cx_ai": int(cx_ai),
                    "cy_ai": int(cy_ai),
                })

            # --- Send pending delivery confirmations (person already left the frame) ---
            for pending_id in list(tracker.pending_post_ids):
                ts = datetime.datetime.now().strftime('%H:%M:%S')
                # Snapshot encoding in the detection thread (frame is already ready, encoding is fast)
                _, _snap_buf = cv2.imencode('.jpg', trace_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                _snap_bytes = _snap_buf.tobytes()
                def send_post_confirmed(pid, snap_bytes, night, timestamp):
                    try:
                        client.publish(MQTT_TOPIC + "/snapshot", snap_bytes, qos=0, retain=True)
                        payload = {
                            "timestamp": timestamp, "label": "persoon", "id": pid,
                            "score": 0, "is_post": True, "post_state": POST_STATE_CONFIRMED,
                            "is_new_detection": False, "is_night": bool(night),
                            "detections": [{
                                "id": pid, "label": "persoon", "score": 0,
                                "is_post": True, "post_state": POST_STATE_CONFIRMED,
                                "is_new_detection": False, "cx_ai": 0, "cy_ai": 0,
                            }],
                        }
                        client.publish(MQTT_TOPIC, json.dumps(payload), qos=0, retain=True)
                        print(f"[{timestamp}] MQTT: ID={pid} Label=persoon State={POST_STATE_CONFIRMED} New=False Post=True")
                    except Exception as e:
                        print(f"MQTT Error: {e}")
                _async_executor.submit(send_post_confirmed, pending_id, _snap_bytes, is_night, ts)
                if pending_id not in tracker.mail_sent_ids:
                    _, _mail_buf = cv2.imencode('.jpg', trace_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    send_post_email(_mail_buf.tobytes())
                    tracker.mail_sent_ids.add(pending_id)
                tracker.pending_post_ids.discard(pending_id)

            # --- MQTT ---
            if found_objects:
                curr_t = time.time()
                if curr_t - last_mqtt_time > MQTT_INTERVAL:
                    last_mqtt_time = curr_t
                    ts = datetime.datetime.now().strftime('%H:%M:%S')

                    primary = found_objects[0]
                    for fo in found_objects:
                        if fo['is_post']:
                            primary = fo
                            break
                        if fo['post_state'] == POST_STATE_MAILBOX:
                            primary = fo
                        elif fo['is_new_detection'] and primary['post_state'] == POST_STATE_IDLE:
                            primary = fo

                    if primary['is_new_detection']:
                        tracker.reported_ids.add(primary['id'])

                    # Snapshot encoding in the detection thread (frame is already ready, encoding is fast)
                    _, _snap_buf = cv2.imencode('.jpg', trace_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    _snap_bytes = _snap_buf.tobytes()

                    def send_mqtt_async(data_list, snap_bytes, night, timestamp, p_obj):
                        try:
                            client.publish(MQTT_TOPIC + "/snapshot", snap_bytes, qos=0, retain=True)
                            payload = {
                                "timestamp": timestamp,
                                "label": p_obj['label'],
                                "id": p_obj['id'],
                                "score": p_obj['score'],
                                "is_post": p_obj['is_post'],
                                "post_state": p_obj['post_state'],
                                "is_new_detection": p_obj['is_new_detection'],
                                "detections": data_list,
                                "is_night": bool(night),
                            }
                            client.publish(MQTT_TOPIC, json.dumps(payload), qos=0, retain=True)
                            print(
                                f"[{timestamp}] MQTT: ID={p_obj['id']} "
                                f"Label={p_obj['label']} State={p_obj['post_state']} "
                                f"New={p_obj['is_new_detection']} Post={p_obj['is_post']}"
                            )
                        except Exception as e:
                            print(f"MQTT Error: {e}")

                    # via executor: only the MQTT publish is async (can occasionally block briefly)
                    _async_executor.submit(
                        send_mqtt_async,
                        list(found_objects), _snap_bytes, is_night, ts, primary
                    )

                    if primary['is_post'] and primary['id'] not in tracker.mail_sent_ids:
                        _, _mail_buf = cv2.imencode('.jpg', trace_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        send_post_email(_mail_buf.tobytes())
                        tracker.mail_sent_ids.add(primary['id'])

            # Night indicator
            if is_night:
                cv2.putText(trace_frame, "NIGHT", (w - 80, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

            with lock:
                output_frame = trace_frame.copy()

        except Exception as e:
            print(f"Loop Error: {e}")
            time.sleep(1)


# =============================================================================
# WEB SERVER
# =============================================================================
def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                frame_copy = None
            else:
                frame_copy = output_frame.copy()
        if frame_copy is None:
            time.sleep(0.05)
            continue
        ret, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return (
        "<html><body style='background:#111; color:#fff; font-family:sans-serif;'>"
        "<h1>🪸 Coral AI — Mailbox Detection</h1>"
        "<img src='/video_feed' width='800' style='border:2px solid #444;'>"
        "<p style='color:#aaa; font-size:13px;'>"
        "Gray=IDLE &nbsp; Yellow=APPROACHING &nbsp; Orange=AT_MAILBOX &nbsp; Red=POST_CONFIRMED"
        "</p></body></html>"
    )


if __name__ == '__main__':
    threading.Thread(target=detect_objects, daemon=True).start()
    serve(app, host='0.0.0.0', port=5000, threads=6)
