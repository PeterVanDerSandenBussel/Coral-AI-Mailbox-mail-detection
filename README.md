# 🪸 Web Detectie — Coral AI Postbode-detectiesysteem

RTSP-camera video-detectiesysteem op basis van een **Coral AI Edge TPU**, dat personen (o.a. de postbode) detecteert bij de brievenbus via een state machine, en de status publiceert naar **MQTT** (Home Assistant), verstuurt via **e-mail**, en toont via een ingebouwde **MJPEG webstream**.

## Inhoud

- [Functionaliteit](#functionaliteit)
- [Architectuur](#architectuur)
- [Postbode state machine](#postbode-state-machine)
- [Vereisten](#vereisten)
- [Installatie](#installatie)
- [Configuratie (config.ini)](#configuratie-configini)
- [Gebruik](#gebruik)
- [MQTT payload](#mqtt-payload)
- [Webinterface](#webinterface)
- [Debug-overlay](#debug-overlay)
- [Belangrijke implementatiedetails](#belangrijke-implementatiedetails)

## Functionaliteit

- Leest een RTSP-camerastream (bv. Hikvision) in een aparte thread met **watchdog** (automatisch herverbinden bij vastlopen).
- Voert objectdetectie uit met een **Coral Edge TPU**-model (via `pycoral`), gefilterd op `target_labels` uit de config (bv. `persoon`).
- Houdt objecten tussen frames bij met een lichtgewicht **centroid tracker** (eigen ID per object, geen zware re-ID-modellen).
- Herkent het **postbode-bezoekpatroon** (aankomst → brievenbus → vertrek) via een aparte state machine per persoon, met valse-positievenfilters (exclude-zone, richtingscheck, dwell-tijd).
- Onderscheidt **dag/nacht** op basis van gemiddelde helderheid van het beeld en gebruikt een lagere confidence-drempel en detectie-interval 's nachts.
- Publiceert detecties + snapshot naar **MQTT** (retained), met throttling via `mqtt.interval`.
- Verstuurt optioneel een **e-mail met foto** zodra een postbezorging bevestigd is.
- Toont een live **MJPEG-videofeed met debug-overlay** via een ingebouwde Flask/Waitress webserver (`/video_feed`, `/`).
- Alle MQTT- en mailacties lopen via een gedeelde `ThreadPoolExecutor` zodat de detectie-loop niet blokkeert.

## Architectuur

```
RTSPReader (thread)          detect_objects() (thread)         Flask/Waitress (main thread)
  └─ leest camera frames  →    ├─ Coral TPU inference               └─ /video_feed (MJPEG stream)
     met watchdog-herstart      ├─ CentroidTracker (ID-toewijzing)   └─ / (HTML preview pagina)
                                ├─ PostTracker (state machine)
                                ├─ MQTT publish (async, executor)
                                └─ E-mail bij bevestiging (async, executor)
```

- **`RTSPReader`** — eigen thread die continu frames leest (throttled op `reader_fps`), met een **watchdog-thread** die de stream herstart als er > `WATCHDOG_SECONDS` (8s) geen nieuw frame binnenkomt. Voorkomt dubbele RTSP-sessies door netjes op de oude thread te wachten (`join()`) voordat een nieuwe start.
- **`CentroidTracker`** — matcht detecties tussen frames op basis van label + afstand tussen centroïdes, kent IDs toe/verwijdert ze na `max_disappeared` gemiste frames, en reset zichzelf volledig als er > `reset_seconds` niets is gezien.
- **`PostTracker`** — losstaande state machine per object-ID die het postbezorgpatroon herkent (zie hieronder).
- **Detectie-loop** (`detect_objects`) — de hoofd-thread die frames ophaalt, inference draait, tracking/state bijwerkt, de debug-overlay tekent en MQTT/mail triggert. Gethrottled op `detect_fps`.
- **Webserver** — Flask app geserveerd via **Waitress** (production WSGI server), toont de laatste frame met overlay als MJPEG-stream.

## Postbode state machine

Alle zone-coördinaten zijn in **AI-coördinaten (0–300)**, de resolutie van het detectiemodel.

| State | Betekenis |
|---|---|
| `IDLE` | Rusttoestand, wacht op aankomst van rechts |
| `APPROACHING` | Persoon beweegt van rechts naar de brievenbus |
| `AT_MAILBOX` | Persoon bevindt zich in de brievenbus-zone |
| `DEPARTING` | Persoon verlaat de brievenbus, richting rechts |
| `POST_CONFIRMED` | Volledig patroon bevestigd → post bezorgd |

**Drie harde voorwaarden voor `POST_CONFIRMED`:**

1. **Aankomst** — track start rechtsboven (`cx > arrive_min_cx`, `cy < arrive_max_cy`), niet in de exclude-zone, en beweegt vervolgens naar de brievenbus (`approach_min_seconds` lang).
2. **Post** — persoon bevindt zich in `mailbox_zone` tussen `mailbox_dwell_min` en `mailbox_dwell_max` seconden.
3. **Vertrek** — persoon verlaat de brievenbus en beweegt terug naar rechts (`cx > depart_min_cx`).

**Valse-positievenfilters:**
- **Exclude-zone** rechtsboven negeert voorbijgangers die niet richting brievenbus lopen.
- **Richtingseis** bij aankomst en vertrek (cx moet af-/toenemen), gebaseerd op de laatste 3 posities in de bewegingsgeschiedenis.
- **Dwell max** — te lang stilstaan bij de brievenbus wordt niet als postbode gezien (reset).
- Als een persoon tijdens `DEPARTING` uit beeld verdwijnt (bv. buiten cameraradius) op de juiste positie, wordt de bezorging alsnog bevestigd (`pending_post_ids`) zodra de tracker het object opruimt.
- Na bevestiging volgt een **cooldown** (`post_cooldown`) voordat dezelfde ID weer kan triggeren.

## Vereisten

- Python 3 met:
  - `opencv-python` (`cv2`)
  - `numpy`
  - `paho-mqtt` (v2 callback API)
  - `flask`
  - `waitress`
  - `pycoral` (Coral Edge TPU runtime + libraries)
- **Coral USB/PCIe Edge TPU** met een gecompileerd `.tflite`-model + labelbestand.
- RTSP-camera (bv. Hikvision) bereikbaar op het netwerk.
- MQTT-broker (bv. Mosquitto, of de Home Assistant-broker).
- (Optioneel) SMTP-account voor e-mailnotificaties.

## Installatie

```bash
pip install opencv-python numpy paho-mqtt flask waitress
# Coral/pycoral installeren volgens de officiële Coral-documentatie
# (Edge TPU runtime + pycoral libraries, afhankelijk van je platform)
```

Plaats `config.ini` in dezelfde map als `web_detectie.py` (zie hieronder).

## Configuratie (config.ini)

Het script leest alle instellingen uit een `config.ini` naast het scriptbestand. Voorbeeldstructuur:

```ini
[camera]
model_path = /pad/naar/model.tflite
label_path = /pad/naar/labels.txt
rtsp_url = rtsp://user:pass@camera-ip:554/stream
reader_fps = 15
detect_fps = 8

[mqtt]
broker = 192.168.1.10
port = 1883
user = mqtt_user
pw = mqtt_password
topic = detectie/brievenbus
interval = 2.0

[mail]
enabled = true
smtp_server = smtp.voorbeeld.nl
smtp_port = 587
from_addr = postdetectie@voorbeeld.nl
from_name = Postdetectie
to_addr = jij@voorbeeld.nl

[detectie]
min_confidence_day = 0.55
min_confidence_night = 0.45
night_interval = 5.0
target_labels = persoon

[post_detectie]
mailbox_zone = 78,130,102,188
exclude_zone = 200,0,300,70
arrive_min_cx = 220
arrive_max_cy = 60
depart_min_cx = 220
depart_max_cy = 60
mailbox_dwell_min = 1.5
mailbox_dwell_max = 15
approach_min_seconds = 1.0
post_cooldown = 300

[tracker]
max_disappeared = 15
reset_seconds = 60
max_trace_points = 30

[debug]
draw_debug = true
draw_traces_debug = true
```

> ⚠️ Bovenstaande waarden zijn illustratief — pas zone-coördinaten aan op basis van je eigen camerabeeld (AI-coördinaten lopen van 0–300, ongeacht de werkelijke cameraresolutie).

**Zone-formaat:** `x1,y1,x2,y2` in AI-coördinaten (0–300), waarbij (0,0) linksboven is.

## Gebruik

```bash
python3 web_detectie.py
```

- De detectie-loop start automatisch in een achtergrondthread.
- De webserver draait op poort **5000** (`0.0.0.0:5000`), via Waitress met 6 threads.
- Live preview: `http://<host>:5000/`
- Ruwe MJPEG-stream: `http://<host>:5000/video_feed`

## MQTT payload

Bij elke detectie (max. 1x per `mqtt.interval` seconden) wordt gepubliceerd op het geconfigureerde topic:

```json
{
  "timestamp": "14:32:10",
  "label": "persoon",
  "id": 3,
  "score": 87,
  "is_post": false,
  "post_state": "APPROACHING",
  "is_new_detection": true,
  "is_night": false,
  "detections": [ /* array van alle gedetecteerde objecten in dit frame */ ]
}
```

Daarnaast wordt op `<topic>/snapshot` een JPEG-snapshot (retained) gepubliceerd bij elke MQTT-update.

Zodra `is_post: true` en `post_state: "POST_CONFIRMED"`, is de bezorging bevestigd — dit is ook het moment waarop (indien ingeschakeld) de e-mail met snapshot wordt verstuurd.

## Webinterface

- **`/`** — eenvoudige HTML-pagina met de live stream en een kleurlegenda:
  - Grijs = `IDLE`
  - Geel = `APPROACHING`
  - Oranje = `AT_MAILBOX`
  - Rood = `POST_CONFIRMED`
- **`/video_feed`** — multipart MJPEG-stream (bruikbaar als camera-bron in Home Assistant of een `<img>`-tag).

## Debug-overlay

Indien `debug.draw_debug = true`:
- Brievenbus-zone en exclude-zone worden als gekleurde vlakken getekend.
- Aankomst-/vertreklijnen (`arrive_min_cx` / `arrive_max_cy`) als referentielijnen.
- Bounding box + ID + label + state per gedetecteerd object, in de kleur van de huidige state.
- "POST BEZORGD!" tekst bij bevestiging.

Indien `debug.draw_traces_debug = true`:
- Bewegingssporen (polylines) per object-ID, gekleurd naar state.

's Nachts verschijnt rechtsboven in beeld een "NACHT"-indicator.

## Belangrijke implementatiedetails

- **Threading-architectuur**: cameralezen, inference/tracking en de webserver draaien in aparte threads; MQTT-publicaties en e-mailverzending lopen via een gedeelde `ThreadPoolExecutor(max_workers=2)` zodat de detectie-loop niet blokkeert op netwerk-I/O.
- **Paho MQTT v2**: client wordt aangemaakt met `CallbackAPIVersion.VERSION2`.
- **Throttling**: aparte FPS-limieten voor het uitlezen van de camera (`reader_fps`, standaard 15) en de detectie/inference-loop (`detect_fps`, standaard 8), instelbaar via `config.ini`.
- **RTSP-stabiliteit**: watchdog herkent een vastgelopen stream (geen nieuw frame > 8s) en herstart de reader-thread pas ná bevestigde afsluiting van de oude thread, om dubbele verbindingen naar de camera te voorkomen (belangrijk bij Hikvision's verbindingslimiet).
- **AI- vs pixelcoördinaten**: alle zonelogica werkt in AI-coördinaten (0–300, het model-inputformaat); `ai_to_px()` / `ai_zone_to_px()` zetten dit om naar pixels voor de overlay op het volledige camerabeeld.
- **Exception-afscherming**: de hoofdloop vangt fouten per iteratie af en logt ze, zodat een incidentele fout (bv. camerahapering) de hele service niet laat crashen.

---

*Dit README.md is automatisch gegenereerd op basis van analyse van `web_detectie.py`. Vul de config.ini-voorbeeldwaarden aan/pas ze aan met je eigen camera-, MQTT- en mailgegevens.*
