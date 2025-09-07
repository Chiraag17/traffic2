# traffic_light_detector.py

import cv2
import numpy as np
import argparse
import time
import os
import math
import csv
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, help="Path to input video file (optional). If omitted, webcam is used.")
args = parser.parse_args()

source = args.video if args.video else 0
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"[ERROR] Could not open video source: {source}")
    exit(1)


if args.video:
    base = os.path.splitext(os.path.basename(args.video))[0]
else:
    base = "webcam"

out_video_fname = f"{base}_output.avi"
report_fname = f"{base}_report.txt"
csv_fname = f"{base}_output.csv"


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0 or np.isnan(fps):
    fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(out_video_fname, fourcc, fps, (width, height))



HSV_RANGES = {
    "red1": (np.array([0, 100, 70]), np.array([10, 255, 255])),
    "red2": (np.array([160, 100, 70]), np.array([180, 255, 255])),
    "yellow": (np.array([15, 100, 120]), np.array([35, 255, 255])),
    "green": (np.array([36, 60, 60]), np.array([90, 255, 255])),
}


MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 20000
CIRCULARITY_MIN = 0.4
CIRCULARITY_MAX = 1.2

# stats
frame_counts = {"RED": 0, "YELLOW": 0, "GREEN": 0, "TOTAL": 0}

# CSV setup
csv_file = open(csv_fname, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "red", "yellow", "green"])  # header

# utility
def morph_mask(mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def combined_red_mask(hsv):
    m1 = cv2.inRange(hsv, HSV_RANGES["red1"][0], HSV_RANGES["red1"][1])
    m2 = cv2.inRange(hsv, HSV_RANGES["red2"][0], HSV_RANGES["red2"][1])
    return cv2.bitwise_or(m1, m2)

def detect_color_blobs(hsv, color_name):
    if color_name == "red":
        mask = combined_red_mask(hsv)
    else:
        lower, upper = HSV_RANGES[color_name]
        mask = cv2.inRange(hsv, lower, upper)
    mask = morph_mask(mask, ksize=5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circularity = 4 * math.pi * (area / (perim * perim))
        if not (CIRCULARITY_MIN <= circularity <= CIRCULARITY_MAX):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        blobs.append({"cnt": cnt, "bbox": (x, y, w, h)})
    return blobs

def draw_blobs(frame, blobs, color_name):
    color_bgr = (0,0,255) if color_name=="red" else (0,255,255) if color_name=="yellow" else (0,255,0)
    for b in blobs:
        x,y,w,h = b["bbox"]
        cv2.rectangle(frame, (x,y), (x+w, y+h), color_bgr, 2)
    if blobs:
        x0,y0,_,_ = blobs[0]["bbox"]
        cv2.putText(frame, f"{color_name.upper()} ({len(blobs)})", (x0, max(0,y0-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# -------------------------
# Main loop
# -------------------------
frame_num = 0
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
    frame_counts["TOTAL"] += 1

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_blobs = detect_color_blobs(hsv, "red")
    yellow_blobs = detect_color_blobs(hsv, "yellow")
    green_blobs = detect_color_blobs(hsv, "green")

    red_detected = 1 if red_blobs else 0
    yellow_detected = 1 if yellow_blobs else 0
    green_detected = 1 if green_blobs else 0

    # update totals
    if red_detected: frame_counts["RED"] += 1
    if yellow_detected: frame_counts["YELLOW"] += 1
    if green_detected: frame_counts["GREEN"] += 1

    # write CSV row
    csv_writer.writerow([frame_num, red_detected, yellow_detected, green_detected])

    # draw
    draw_blobs(frame, red_blobs, "red")
    draw_blobs(frame, yellow_blobs, "yellow")
    draw_blobs(frame, green_blobs, "green")

    # FPS
    cur = time.time()
    fps_disp = 1.0 / (cur - prev_time) if cur != prev_time else 0.0
    prev_time = cur
    cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    cv2.imshow("Traffic Light Detector", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
csv_file.close()

# -------------------------
# Save report
# -------------------------
total = frame_counts["TOTAL"]
lines = []
lines.append("Traffic Light Detection Report")
lines.append("="*36)
lines.append(f"Generated at: {datetime.now().isoformat()}")
lines.append(f"Source: {source}")
lines.append(f"Total frames processed: {total}\n")
for c in ("RED","YELLOW","GREEN"):
    cnt = frame_counts[c]
    pct = (cnt / total * 100) if total>0 else 0.0
    lines.append(f"{c}: {cnt} frames ({pct:.2f}%)")
lines.append("")
lines.append(f"Output video: {out_video_fname}")
lines.append(f"CSV log: {csv_fname}")

report_text = "\n".join(lines)
print("\n" + report_text)

with open(report_fname, "w") as f:
    f.write(report_text)

print(f"\nReport saved to: {report_fname}")
print(f"CSV log saved to: {csv_fname}")
print(f"Annotated video saved to: {out_video_fname}")
