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

# --- 1. CONFIGURATIE (gelezen uit config.ini) ---
_cfg = configparser.ConfigParser()
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
if not _cfg.read(_cfg_path):
    raise FileNotFoundError(f"config.ini niet gevonden op {_cfg_path}")

MODEL_PATH = _cfg.get('camera', 'model_path')
LABEL_PATH = _cfg.get('camera', 'label_path')
RTSP_URL   = _cfg.get('camera', 'rtsp_url')

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
# POST-DETECTIE CONFIGURATIE
#
# Alle zones in AI-coordinaten (0-300), want cx_ai/cy_ai zijn AI-coordinaten.
#
# Postbode-patroon (uit logging):
#   AANKOMST : van rechtsboven  cx > ARRIVE_MIN_CX, cy < ARRIVE_MAX_CY
#   BRIEVENBUS: cx ~ 78-102, cy ~ 130-188  MAILBOX_ZONE_AI
#   VERTREK  : terug naar rechtsboven       cx > DEPART_MIN_CX
#
# EXCLUDE_ZONE_AI: gebied rechtsboven waar voorbijgangers lopen
#   AI [200,0] t/m [300,70] = rechtsboven hoek, voorbijgangerspad
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

TRACKER_MAX_DISAPPEARED  = _cfg.getint('tracker', 'max_disappeared')
TRACKER_RESET_SECONDS    = _cfg.getint('tracker', 'reset_seconds')
TRACKER_MAX_TRACE_POINTS = _cfg.getint('tracker', 'max_trace_points')

# debug-overlay
DRAW_DEBUG        = _cfg.getboolean('debug', 'draw_debug')
DRAW_TRACES_DEBUG = _cfg.getboolean('debug', 'draw_traces_debug')

# Camera belasting: maximale framerate voor de reader-thread en detectie-loop.
# Hikvision levert doorgaans 25fps; 8fps is ruim voldoende voor loopdetectie.
# Pas ook aan in config.ini als gewenst: [camera] reader_fps / detect_fps
READER_FPS  = _cfg.getfloat('camera', 'reader_fps')   if _cfg.has_option('camera', 'reader_fps')  else 15.0
DETECT_FPS  = _cfg.getfloat('camera', 'detect_fps')   if _cfg.has_option('camera', 'detect_fps')  else  8.0
_READER_INTERVAL = 1.0 / READER_FPS   # seconden tussen reads
_DETECT_INTERVAL = 1.0 / DETECT_FPS   # seconden tussen detecties

output_frame = None
lock = threading.Lock()
is_night = False

app = Flask(__name__)

# FIX 2: gedeelde thread-pool voor alle MQTT/mail async taken (max 2 workers)
_async_executor = ThreadPoolExecutor(max_workers=2)

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id="")
client.username_pw_set(MQTT_USER, MQTT_PW)
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()


# =============================================================================
# E-MAIL FUNCTIE
# =============================================================================
def send_post_email(jpeg_bytes):
    """Stuurt een e-mail met snapshot bij post detectie. Draait via executor."""
    if not MAIL_ENABLED:
        return
    def _send(data):
        try:
            ts = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            msg = MIMEMultipart()
            msg['Subject'] = f"Post bezorgd! 📬 {ts}"
            msg['From']    = f"{MAIL_FROM_NAME} <{MAIL_FROM}>"
            msg['To']      = MAIL_TO
            msg.attach(MIMEText(f"De postbode is gedetecteerd op {ts}.", 'plain'))
            msg.attach(MIMEImage(data, name="post_snapshot.jpg"))
            with smtplib.SMTP(MAIL_SMTP_SERVER, MAIL_SMTP_PORT, timeout=10) as smtp:
                smtp.sendmail(MAIL_FROM, MAIL_TO, msg.as_string())
            print(f"[MAIL] Verstuurd naar {MAIL_TO} om {ts}")
        except Exception as e:
            print(f"[MAIL] Fout: {e}")
    _async_executor.submit(_send, jpeg_bytes)


# =============================================================================
# HULPFUNCTIES: zone-check in AI-coordinaten
# =============================================================================
def point_in_zone_ai(cx_ai, cy_ai, zone_ai):
    """Controleert of een AI-centroide binnen een AI-zone valt."""
    return zone_ai[0] <= cx_ai <= zone_ai[2] and zone_ai[1] <= cy_ai <= zone_ai[3]

def ai_to_px(ai_x, ai_y, frame_w, frame_h):
    """Zet AI-coordinaten (300x300) om naar pixel-coordinaten voor tekenen."""
    size = min(frame_h, frame_w)
    start_x = (frame_w - size) // 2
    start_y = (frame_h - size) // 2
    px_x = int(ai_x * size / 300.0) + start_x
    px_y = int(ai_y * size / 300.0) + start_y
    return px_x, px_y

def ai_zone_to_px(zone_ai, frame_w, frame_h):
    """Zet AI-zone om naar pixel-rechthoek voor tekenen."""
    x1, y1 = ai_to_px(zone_ai[0], zone_ai[1], frame_w, frame_h)
    x2, y2 = ai_to_px(zone_ai[2], zone_ai[3], frame_w, frame_h)
    return x1, y1, x2, y2


# =============================================================================
# STATE MACHINE PER PERSOON
#
# Drie harde voorwaarden voor POST_CONFIRMED:
#   1. AANKOMST  - track start rechts (cx > ARRIVE_MIN_CX), niet in exclude-zone,
#                  en beweegt naar linksonder richting brievenbus
#   2. POST      - persoon in MAILBOX_ZONE_AI tussen MAILBOX_DWELL_MIN en MAILBOX_DWELL_MAX sec
#   3. VERTREK   - persoon verlaat mailbox en beweegt terug naar rechts (cx > DEPART_MIN_CX)
#
# Valse vlaggen worden voorkomen door:
#   - Exclude zone rechtsboven: voorbijgangers die niet naar brievenbus gaan
#   - Richtingseis aankomst: cx moet afnemen op weg naar brievenbus
#   - Richtingseis vertrek: cx moet toenemen na brievenbus
#   - Dwell max: te lang stilstaan = geen postbode
# =============================================================================
POST_STATE_IDLE      = "IDLE"
POST_STATE_APPROACH  = "APPROACHING"
POST_STATE_MAILBOX   = "AT_MAILBOX"
POST_STATE_DEPARTING = "DEPARTING"
POST_STATE_CONFIRMED = "POST_CONFIRMED"


class PostTracker:
    """Bijhoudt de postbezorgstatus per persoon-ID met richtingsdetectie."""
    def __init__(self):
        self.states        = {}   # obj_id -> state
        self.approach_entry= {}   # obj_id -> timestamp aankomst approach
        self.mailbox_entry = {}   # obj_id -> timestamp aankomst mailbox
        self.confirmed_time= {}   # obj_id -> timestamp bevestiging
        self.first_cx      = {}   # obj_id -> eerste cx (voor richtingscheck aankomst)
        self.cx_history    = {}   # obj_id -> lijst van recente cx waarden
        self.arrived_from_right = set()  # IDs die aantoonbaar van rechts kwamen

    def get_state(self, obj_id):
        return self.states.get(obj_id, POST_STATE_IDLE)

    def _in_exclude(self, cx_ai, cy_ai):
        return point_in_zone_ai(cx_ai, cy_ai, EXCLUDE_ZONE_AI)

    def _moving_left(self, obj_id, cx_ai):
        """Beweegt de persoon naar links (cx neemt af)?"""
        hist = self.cx_history.get(obj_id, [])
        if len(hist) < 3:
            return False
        avg_prev = sum(hist[-3:]) / 3
        return cx_ai < avg_prev - 2

    def _moving_right(self, obj_id, cx_ai):
        """Beweegt de persoon naar rechts (cx neemt toe)?"""
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

        # Bijhouden cx-geschiedenis voor richtingsdetectie
        if obj_id not in self.cx_history:
            self.cx_history[obj_id] = []
        self.cx_history[obj_id].append(cx_ai)
        if len(self.cx_history[obj_id]) > 10:
            self.cx_history[obj_id].pop(0)

        # Eerste cx vastleggen
        if obj_id not in self.first_cx:
            self.first_cx[obj_id] = cx_ai

        print(f"[ZONE] ID={obj_id} cx={cx_ai:.0f} cy={cy_ai:.0f} "
              f"mailbox={in_mailbox} excl={in_exclude} state={current_state}")

        # --- Cooldown na bevestiging ---
        if current_state == POST_STATE_CONFIRMED:
            if now - self.confirmed_time.get(obj_id, 0) > POST_COOLDOWN_SECONDS:
                self._reset(obj_id)
            return self.states.get(obj_id, POST_STATE_IDLE), True

        # =========================================================
        # FASE 3 eerst: DEPARTING - check of persoon rechts vertrekt
        # =========================================================
        if current_state == POST_STATE_DEPARTING:
            if not in_mailbox and cx_ai > DEPART_MIN_CX and cy_ai < DEPART_MAX_CY:
                self._confirm(obj_id, now)
                return POST_STATE_CONFIRMED, True
            return POST_STATE_DEPARTING, False

        # =========================================================
        # FASE 2: IN de mailbox
        # =========================================================
        elif current_state == POST_STATE_MAILBOX:
            if in_mailbox:
                dwell = now - self.mailbox_entry.get(obj_id, now)
                if dwell > MAILBOX_DWELL_MAX:
                    print(f"[POST] ID={obj_id} -> RESET (dwell te lang: {dwell:.1f}s)")
                    self._reset(obj_id)
                elif dwell >= MAILBOX_DWELL_MIN:
                    self.states[obj_id] = POST_STATE_DEPARTING
                    print(f"[POST] ID={obj_id} -> DEPARTING (dwell={dwell:.1f}s)")
            else:
                dwell = now - self.mailbox_entry.get(obj_id, now)
                if dwell < MAILBOX_DWELL_MIN:
                    print(f"[POST] ID={obj_id} -> terug APPROACH (dwell te kort: {dwell:.1f}s)")
                    self.states[obj_id] = POST_STATE_APPROACH
                else:
                    self.states[obj_id] = POST_STATE_DEPARTING
                    print(f"[POST] ID={obj_id} -> DEPARTING (verliet mailbox na {dwell:.1f}s)")

        # =========================================================
        # FASE 1b: APPROACHING - op weg naar brievenbus
        # =========================================================
        elif current_state == POST_STATE_APPROACH:
            if in_exclude:
                print(f"[POST] ID={obj_id} -> RESET (in exclude zone)")
                self._reset(obj_id)
            elif in_mailbox:
                approach_time = now - self.approach_entry.get(obj_id, now)
                if approach_time >= APPROACH_MIN_SECONDS and obj_id in self.arrived_from_right:
                    self.mailbox_entry[obj_id] = now
                    self.states[obj_id] = POST_STATE_MAILBOX
                    print(f"[POST] ID={obj_id} -> AT_MAILBOX (approach={approach_time:.1f}s)")
                else:
                    reason = "approach te kort" if approach_time < APPROACH_MIN_SECONDS else "niet van rechts"
                    print(f"[POST] ID={obj_id} -> RESET ({reason})")
                    self._reset(obj_id)

        # =========================================================
        # FASE 1a: IDLE - wacht op aankomst van rechts
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
        print(f"[POST] ID={obj_id} POST BEVESTIGD op {datetime.datetime.now().strftime('%H:%M:%S')}")

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

    def update(self, rects, labels_list):
        current_time = time.time()
        if (current_time - self.last_seen_time) > self.reset_after_seconds:
            self.__init__(self.max_disappeared, self.reset_after_seconds, self.max_trace_points)

        if not rects:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    state = self._post_tracker.get_state(obj_id)
                    if state == POST_STATE_DEPARTING:
                        last_pos = self.objects.get(obj_id, (0, 0))
                        last_cx, last_cy = last_pos[0], last_pos[1]
                        if last_cx > DEPART_MIN_CX and last_cy < DEPART_MAX_CY:
                            print(f"[POST] ID={obj_id} verdwenen rechtsboven tijdens DEPARTING -> POST BEVESTIGD")
                            self._post_tracker._confirm(obj_id, time.time())
                            self.pending_post_ids.add(obj_id)
                        else:
                            print(f"[POST] ID={obj_id} verdwenen maar positie cx={last_cx:.0f} cy={last_cy:.0f} klopt niet, geen post")
                    self._post_tracker.remove(obj_id)
                    self.deregister(obj_id)
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

            for i, i_centroid in enumerate(input_centroids):
                i_label = labels_list[i]
                distances, valid_ids = [], []
                for oid in object_ids:
                    if oid in registered_this_frame:
                        continue
                    if self.labels[oid] == i_label:
                        d = math.hypot(i_centroid[0] - self.objects[oid][0],
                                       i_centroid[1] - self.objects[oid][1])
                        distances.append(d)
                        valid_ids.append(oid)

                if distances and min(distances) < 60:
                    idx = distances.index(min(distances))
                    best_id = valid_ids[idx]
                    self.objects[best_id] = i_centroid
                    self.disappeared[best_id] = 0
                    registered_this_frame.add(best_id)

                    if best_id not in self.traces:
                        self.traces[best_id] = []
                    self.traces[best_id].append(i_centroid)
                    if len(self.traces[best_id]) > self.max_trace_points:
                        self.traces[best_id].pop(0)
                else:
                    self.register(i_centroid, i_label)

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
# STATE-KLEUREN voor overlay
# =============================================================================
STATE_COLORS = {
    POST_STATE_IDLE:      (180, 180, 180),  # Grijs
    POST_STATE_APPROACH:  (0, 200, 255),    # Geel-oranje
    POST_STATE_MAILBOX:   (0, 165, 255),    # Oranje
    POST_STATE_DEPARTING: (0, 100, 255),    # Oranje-rood
    POST_STATE_CONFIRMED: (0, 0, 255),      # Rood (post bevestigd!)
}


# =============================================================================
# RTSP CAMERA READER - aparte thread met watchdog
# FIX 1: watchdog wacht met thread.join() tot oude reader-thread echt gestopt is
#         voordat een nieuwe wordt gestart. Voorkomt twee gelijktijdige cap.read()
#         sessies naar de camera (connection limit op Hikvision).
# =============================================================================
class RTSPReader:
    WATCHDOG_SECONDS = 8  # Na N seconden geen nieuw frame -> herverbinden

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
                # Throttle: wacht rest van het frame-interval zodat we de camera
                # niet sneller bevragen dan READER_FPS (standaard 15 fps).
                elapsed = time.time() - t0
                wait = _READER_INTERVAL - elapsed
                if wait > 0:
                    time.sleep(wait)
            else:
                print("[CAM] cap.read() mislukt -- herverbinden...")
                cap.release()
                time.sleep(3)
                if not self._stop_reader:
                    cap = self._make_cap()
        cap.release()
        print("[CAM] Reader thread gestopt.")

    def _watchdog_loop(self):
        """Detecteert vastgelopen stream. FIX 1: join() wacht tot oude thread klaar is."""
        while True:
            time.sleep(3)
            if self._last_frame_time == 0:
                continue
            age = time.time() - self._last_frame_time
            if age > self.WATCHDOG_SECONDS:
                print(f"[CAM] Watchdog: geen frame sinds {age:.0f}s -- stream herstarten")
                # FIX 1: stop oude thread en wacht tot hij echt klaar is
                self._stop_reader = True
                self._thread.join(timeout=5)  # max 5s wachten
                if self._thread.is_alive():
                    print("[CAM] Waarschuwing: oude reader thread nog actief na 5s timeout")
                # Nu pas nieuwe thread starten (geen dubbele RTSP-verbinding meer)
                self._stop_reader = False
                self._last_frame_time = time.time()  # voorkom directe herhaling
                self._thread = threading.Thread(target=self._reader_loop, daemon=True)
                self._thread.start()
                print("[CAM] Reader thread herstart.")

    def read(self):
        """Geeft (True, frame) of (False, None) -- nooit blokkerend."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()


# =============================================================================
# HOOFD DETECTIE-LOOP
# =============================================================================
def detect_objects():
    global output_frame, is_night
    last_mqtt_time = 0
    last_grey_time = 0

    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    labels = read_label_file(LABEL_PATH)

    camera = RTSPReader(RTSP_URL)
    print("[CAM] Wachten op eerste frame...")
    while True:
        ok, _ = camera.read()
        if ok:
            break
        time.sleep(0.2)
    print("[CAM] Stream actief.")

    tracker = CentroidTracker()

    last_detect_time = 0.0

    while True:
        try:
            # Throttle detectie-loop tot DETECT_FPS (standaard 8 fps).
            # Voorkomt dat de Coral + frame-resizing onnodig snel draait.
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
                cv2.putText(trace_frame, "BRIEVENBUS", (mx1 + 2, my1 - 6),
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

            # --- Verwerk gedetecteerde objecten ---
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
                        cv2.putText(trace_frame, "POST BEZORGD!", (20, 50),
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

            # --- Verstuur pending post bevestigingen (persoon al weg uit beeld) ---
            for pending_id in list(tracker.pending_post_ids):
                ts = datetime.datetime.now().strftime('%H:%M:%S')
                # Snapshot encoding in de detectie-thread (frame is al klaar, encode is snel)
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
                        print(f"MQTT Fout: {e}")
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

                    # Snapshot encoding in de detectie-thread (frame is al klaar, encode is snel)
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
                            print(f"MQTT Fout: {e}")

                    # via executor: alleen de MQTT publish is async (kan soms brief blokkeren)
                    _async_executor.submit(
                        send_mqtt_async,
                        list(found_objects), _snap_bytes, is_night, ts, primary
                    )

                    if primary['is_post'] and primary['id'] not in tracker.mail_sent_ids:
                        _, _mail_buf = cv2.imencode('.jpg', trace_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        send_post_email(_mail_buf.tobytes())
                        tracker.mail_sent_ids.add(primary['id'])

            # Nacht-indicator
            if is_night:
                cv2.putText(trace_frame, "NACHT", (w - 80, 25),
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
        "<h1>🪸 Coral AI — Post Detectie</h1>"
        "<img src='/video_feed' width='800' style='border:2px solid #444;'>"
        "<p style='color:#aaa; font-size:13px;'>"
        "Grijs=IDLE &nbsp; Geel=APPROACHING &nbsp; Oranje=AT_MAILBOX &nbsp; Rood=POST_CONFIRMED"
        "</p></body></html>"
    )


if __name__ == '__main__':
    threading.Thread(target=detect_objects, daemon=True).start()
    serve(app, host='0.0.0.0', port=5000, threads=6)
