"""
AI Fight Detection — FINAL VERSION
====================================
APPROACH:
  - YOLOv8-Pose: ONE model handles person detection + skeleton (no MediaPipe)
  - Bounding box overlap (IoU) for collision — reliable, no tracking gaps
  - Skeleton drawn from YOLO keypoints for visual appeal
  - 640x360 resolution, process every 4 frames — smooth on any laptop
  - Grace period: 1.5s of no overlap before reset — no more timer flickering
  - Screenshots: tested at startup, saves on warning + every 3s after (max 5)
  - CSV: logs every event properly
  - Unlimited people: checks every pair

CAMERA: http://172.20.10.4:8080/video
RUN:    python fight_detection.py
KEYS:   Q = quit | S = manual screenshot
"""

import cv2, numpy as np, time, csv, os
from datetime import datetime
from itertools import combinations

# ── Sound ─────────────────────────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    SOUND_OK = True
except:
    SOUND_OK = False
    print("⚠  No sound — pip install pygame")

# ── YOLOv8-Pose ───────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n-pose.pt")   # auto-downloads ~7MB, detects people + skeleton
    print("✅ YOLOv8-Pose ready")
except Exception as e:
    print(f"❌ YOLO failed: {e}"); exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
STREAM          = "http://172.20.10.4:8080/video"
W, H            = 854, 480         # display resolution
PROC_EVERY      = 4                # run AI every N frames
IOU_THRESHOLD   = 0.05             # bounding box overlap % to count as collision
COLLISION_SECS  = 5                # seconds of overlap → WARNING
ALARM_MINS      = 15               # minutes of warning → ALARM
GRACE_SECS      = 1.5              # seconds of no overlap before reset
MAX_PHOTOS      = 5                # photos per fight
PHOTO_GAP       = 3.0              # seconds between auto photos
LOG_FILE        = "fight_log.csv"
SHOT_DIR        = "screenshots"

# COCO skeleton connections for drawing
SKEL = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),
        (5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
        (11,12),(11,13),(13,15),(12,14),(14,16)]

COLORS = [(0,220,100),(0,160,255),(220,140,0),(180,0,220),(0,220,220)]
RED    = (30, 30,220)
YELLOW = (0, 215,255)
WHITE  = (240,240,240)
DARK   = (18, 18, 30)

# ── Collision: bounding box IoU ───────────────────────────────────────────────
def iou(b1, b2):
    """Intersection over Union of two boxes (x1,y1,x2,y2)."""
    ix1 = max(b1[0],b2[0]); iy1 = max(b1[1],b2[1])
    ix2 = min(b1[2],b2[2]); iy2 = min(b1[3],b2[3])
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    if inter == 0: return 0.0
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / float(a1 + a2 - inter)

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_person(frame, box, kpts, color, label):
    x1,y1,x2,y2 = box
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    cv2.putText(frame,label,(x1+4,y1+20),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    if kpts is None or len(kpts) < 17:
        return

    # Draw skeleton lines
    for a,b in SKEL:
        if kpts[a][2]>0.4 and kpts[b][2]>0.4:
            pa = (int(kpts[a][0]),int(kpts[a][1]))
            pb = (int(kpts[b][0]),int(kpts[b][1]))
            cv2.line(frame,pa,pb,color,2)

    # Draw joints
    for k in kpts:
        if k[2]>0.4:
            cv2.circle(frame,(int(k[0]),int(k[1])),4,color,-1)

def draw_hud(frame, s, now):
    fh, fw = frame.shape[:2]

    # Top bar
    bar = RED if s["warn"] else DARK
    cv2.rectangle(frame,(0,0),(fw,50),bar,-1)
    title = "!! FIGHT DETECTED — ALERT !!" if s["warn"] else "AI SURVEILLANCE — MONITORING"
    cv2.putText(frame,title,(12,22),cv2.FONT_HERSHEY_DUPLEX,0.65,WHITE,1)
    cv2.putText(frame,datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (12,42),cv2.FONT_HERSHEY_SIMPLEX,0.4,(180,180,200),1)

    # REC blink
    if int(now*2)%2==0:
        cv2.circle(frame,(fw-28,18),7,(0,0,220),-1)
        cv2.putText(frame,"REC",(fw-55,23),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,220),1)

    # Bottom bar
    cv2.rectangle(frame,(0,fh-50),(fw,fh),DARK,-1)
    cv2.line(frame,(0,fh-50),(fw,fh-50),(50,50,70),1)

    pc = s["persons"]
    pc_color = (0,220,100) if pc>=2 else YELLOW
    cv2.putText(frame,f"People: {pc}",
                (12,fh-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,pc_color,1)

    if s["colliding"] and s["coll_start"]:
        if s["warn"]:
            left = max(0, ALARM_MINS*60-(now-s["warn_start"]))
            cv2.putText(frame,f"Alarm in {int(left//60):02d}:{int(left%60):02d}",
                        (12,fh-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,RED,1)
            cv2.putText(frame,f"Photos: {s['photos']}/{MAX_PHOTOS}  Alerts: {s['alerts']}",
                        (fw-280,fh-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,YELLOW,1)
        else:
            left = max(0,COLLISION_SECS-(now-s["coll_start"]))
            cv2.putText(frame,f"Bodies overlapping — Alert in {left:.1f}s",
                        (12,fh-10),cv2.FONT_HERSHEY_SIMPLEX,0.48,YELLOW,1)
    else:
        cv2.putText(frame,"Monitoring — No collision",
                    (12,fh-10),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,220,100),1)
        cv2.putText(frame,f"Alerts: {s['alerts']}  Photos: {s['total_photos']}",
                    (fw-250,fh-10),cv2.FONT_HERSHEY_SIMPLEX,0.42,(130,130,150),1)

    if s["warn"] and int(now*2)%2==0:
        cv2.rectangle(frame,(0,0),(fw,fh),RED,5)

    if s["alarm"]:
        cv2.putText(frame,"!! ALARM SOUNDING !!",
                    (fw//2-170,fh//2),cv2.FONT_HERSHEY_DUPLEX,1.1,YELLOW,3)

    cv2.putText(frame,"Q=Quit  S=Screenshot",
                (fw-185,fh-30),cv2.FONT_HERSHEY_SIMPLEX,0.38,(60,60,80),1)

# ── Sound ─────────────────────────────────────────────────────────────────────
def beep(freq=880,ms=400):
    if not SOUND_OK: return
    try:
        sr=44100; t=np.linspace(0,ms/1000,int(sr*ms/1000),False)
        w=(np.sin(2*np.pi*freq*t)*32767).astype(np.int16)
        pygame.sndarray.make_sound(np.column_stack([w,w])).play()
    except: pass

# ── CSV ───────────────────────────────────────────────────────────────────────
def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE,"w",newline="") as f:
            csv.writer(f).writerow(
                ["Timestamp","Event","Persons_Involved","Duration_sec","Photos_Saved"])

def log_ev(event, persons="", dur=0, photos=0):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE,"a",newline="") as f:
        csv.writer(f).writerow([ts,event,persons,f"{dur:.1f}",photos])
    print(f"📝 [{ts}] {event}  {persons}")

# ── Screenshot ────────────────────────────────────────────────────────────────
def save_photo(frame, n, fid):
    path = os.path.join(os.path.abspath(SHOT_DIR), f"fight_{fid}_{n}.jpg")
    ok   = cv2.imwrite(path, frame)
    if ok: print(f"📸 Photo {n}/{MAX_PHOTOS} → {path}")
    else:  print(f"❌ FAILED to save photo → {path}")
    return ok

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SHOT_DIR, exist_ok=True)
    init_log()

    # Test screenshot works at startup
    test = os.path.join(os.path.abspath(SHOT_DIR),"_test.jpg")
    cv2.imwrite(test, np.zeros((10,10,3),dtype=np.uint8))
    if os.path.exists(test):
        print(f"✅ Screenshots will save to: {os.path.abspath(SHOT_DIR)}")
        os.remove(test)
    else:
        print(f"❌ CANNOT save screenshots to {SHOT_DIR} — check folder permissions!")

    print("="*55)
    print("  AI FIGHT DETECTION — FINAL VERSION")
    print("="*55)
    print(f"  Camera   : {STREAM}")
    print(f"  Warning  : after {COLLISION_SECS}s overlap")
    print(f"  Photos   : up to {MAX_PHOTOS} per fight")
    print(f"  Resolution: {W}x{H}")
    print("="*55+"\n")

    cap = cv2.VideoCapture(STREAM)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ Cannot connect — open in browser to test: {STREAM}"); return

    print("✅ Camera connected!\n")
    log_ev("SYSTEM_START")

    # State
    s = dict(persons=0, colliding=False, coll_start=None,
             last_coll_t=None, warn=False, warn_start=None,
             alarm=False, alerts=0, photos=0, total_photos=0)

    # Cache
    cache_boxes = []   # list of (x1,y1,x2,y2)
    cache_kpts  = []   # list of kpts arrays (17x3) or None
    fc          = 0
    last_beep   = 0
    last_photo  = 0
    fid         = datetime.now().strftime("%H%M%S")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.2)
            cap.release()
            cap = cv2.VideoCapture(STREAM)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        frame = cv2.resize(cv2.flip(frame,1),(W,H))
        now   = time.time()
        fc   += 1

        # ── Run YOLOv8-Pose every PROC_EVERY frames ──
        if fc % PROC_EVERY == 0:
            cache_boxes = []
            cache_kpts  = []
            results = model(frame, conf=0.5, verbose=False)[0]

            if results.boxes is not None:
                for i, box in enumerate(results.boxes):
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cache_boxes.append((x1,y1,x2,y2))
                    # Extract keypoints if available
                    if (results.keypoints is not None and
                        i < len(results.keypoints.data)):
                        kp = results.keypoints.data[i].cpu().numpy()  # 17x3
                        cache_kpts.append(kp)
                    else:
                        cache_kpts.append(None)

        # ── Collision check — bounding box IoU ──
        n = len(cache_boxes)
        colliding_pairs = []
        if n >= 2:
            for i,j in combinations(range(n),2):
                if iou(cache_boxes[i], cache_boxes[j]) >= IOU_THRESHOLD:
                    colliding_pairs.append((i,j))

        colliding_now = len(colliding_pairs) > 0
        hit_set       = set(p for pair in colliding_pairs for p in pair)

        # ── Draw all people ──
        for i,(x1,y1,x2,y2) in enumerate(cache_boxes):
            col   = RED if i in hit_set else COLORS[i % len(COLORS)]
            kpts  = cache_kpts[i] if i < len(cache_kpts) else None
            label = f"P{i+1}"
            draw_person(frame,(x1,y1,x2,y2),kpts,col,label)

        # ── State machine ──
        if colliding_now:
            s["last_coll_t"] = now

            if not s["coll_start"]:
                s["coll_start"] = now
                s["photos"]     = 0
                fid = datetime.now().strftime("%H%M%S")
                pairs = " & ".join([f"P{a+1}↔P{b+1}" for a,b in colliding_pairs])
                print(f"⚠  Overlap: {pairs} — 5s timer started")

            elapsed = now - s["coll_start"]

            # ── 5 seconds → WARNING ──
            if elapsed >= COLLISION_SECS and not s["warn"]:
                s["warn"]      = True
                s["warn_start"]= now
                s["alerts"]   += 1
                pairs = " & ".join([f"P{a+1}+P{b+1}" for a,b in colliding_pairs])
                log_ev("FIGHT_WARNING", persons=pairs)
                beep(1000,600)
                print(f"🚨 WARNING! Taking photo 1 now...")

                # Photo 1 — immediately
                if save_photo(frame, 1, fid):
                    s["photos"]       = 1
                    s["total_photos"] += 1
                    last_photo        = now

            # Photos 2-5 every PHOTO_GAP seconds
            if s["warn"] and s["photos"] < MAX_PHOTOS:
                if now - last_photo >= PHOTO_GAP:
                    n_ph = s["photos"] + 1
                    if save_photo(frame, n_ph, fid):
                        s["photos"]       = n_ph
                        s["total_photos"] += 1
                        last_photo        = now

            # 15 minutes → ALARM
            if s["warn"] and not s["alarm"]:
                if now - s["warn_start"] >= ALARM_MINS * 60:
                    s["alarm"] = True
                    log_ev("ALARM_STARTED")
                    print("🔔 ALARM!")

        else:
            # Reset only after GRACE_SECS of no collision
            if s["last_coll_t"] and (now - s["last_coll_t"]) >= GRACE_SECS:
                if s["coll_start"]:
                    dur = now - s["coll_start"]
                    if s["warn"]:
                        log_ev("FIGHT_RESOLVED", dur=dur, photos=s["photos"])
                        print(f"✅ Fight resolved — {dur:.1f}s | {s['photos']} photos saved")
                    else:
                        print(f"ℹ  Brief contact ended ({dur:.1f}s) — no alert")
                    s.update(coll_start=None, last_coll_t=None,
                             warn=False, warn_start=None, alarm=False)
                    last_photo = 0

        # Alarm beep every 2s
        if s["alarm"] and now - last_beep >= 2:
            beep(1200,500)
            last_beep = now

        s["persons"]   = n
        s["colliding"] = colliding_now
        draw_hud(frame, s, now)

        cv2.imshow("AI Fight Detection — Final", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            p = os.path.join(os.path.abspath(SHOT_DIR),
                f"manual_{datetime.now().strftime('%H%M%S')}.jpg")
            ok = cv2.imwrite(p, frame)
            print(f"📸 Manual screenshot {'saved' if ok else 'FAILED'} → {p}")

    cap.release()
    cv2.destroyAllWindows()
    log_ev("SYSTEM_STOP", photos=s["total_photos"])
    print(f"\n📁 Log    : {os.path.abspath(LOG_FILE)}")
    print(f"📁 Photos : {os.path.abspath(SHOT_DIR)}")

if __name__ == "__main__":
    main()
