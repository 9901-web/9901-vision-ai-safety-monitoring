import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import numpy as np
import math
import csv
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 1. 설정 및 초기화
# ==========================================
VIDEO_PATH = "kimhyunsoo.mp4" 
MODEL_PATH = "yolov8n.pt" 

PERSON_CLS = 0
MACHINERY_CLASSES = [2, 3, 5, 7] 

# [수정] 타원 반지름 대신 다각형(육각형) 꼭짓점 좌표 설정
# 장비의 중심점(0, 0)을 기준으로 한 상대 좌표 [x, y]
WORK_POLYGON = np.array([
    [-150, -100],  # 좌측 상단
    [ 150, -100],  # 우측 상단
    [ 220,    0],  # 우측 중앙
    [ 150,  100],  # 우측 하단
    [-150,  100],  # 좌측 하단
    [-220,    0]   # 좌측 중앙
], np.int32)

HELMET_THRESHOLD = 0.01 

PIXEL_TO_METER = 0.05 
SPEED_THRESHOLD_KMH = 2.5 
MIN_PIXEL_MOVE = 3        

CSV_LOG_PATH = "safety_event_log.csv"
LOG_COOLDOWN_SEC = 2.0 

def check_helmet(frame, x1, y1, x2, y2):
    head_y2 = y1 + int((y2 - y1) * 0.30)
    head_roi = frame[y1:head_y2, x1:x2]
    
    if head_roi.size == 0:
        return True 
        
    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    lower_color = np.array([5, 80, 80])
    upper_color = np.array([40, 255, 255])
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 45, 255])
    
    mask_color = cv2.inRange(hsv, lower_color, upper_color)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_helmet = cv2.bitwise_or(mask_color, mask_white)
    
    ratio = cv2.countNonZero(mask_helmet) / (head_roi.shape[0] * head_roi.shape[1] + 1e-6)
    return ratio > HELMET_THRESHOLD

def write_log(event_type, description, video_time_str):
    file_exists = os.path.isfile(CSV_LOG_PATH)
    with open(CSV_LOG_PATH, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["실제 기록 시간", "영상 내 시간", "이벤트 종류", "상세 내용"])
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([now_str, video_time_str, event_type, description])

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ '{VIDEO_PATH}' 영상을 불러올 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps):
        fps = 30.0

    model = YOLO(MODEL_PATH)
    frame_count = 0 
    track_history = {}
    
    last_log_time = {"DANGER": -LOG_COOLDOWN_SEC, "WARNING": -LOG_COOLDOWN_SEC}
    
    recent_events = []
    MAX_EVENTS_DISPLAY = 4

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if frame_count % 5 not in (1, 3):
            continue

        overlay = frame.copy()
        height, width = frame.shape[:2]

        results = model.track(frame, conf=0.25, persist=True, verbose=False)
        
        persons = []
        valid_machineries = []
        ignored_machineries = []

        boxes = results[0].boxes
        if boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
        else:
            ids = [None] * len(boxes)

        for box, obj_id in zip(boxes, ids):
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            if cls_id == PERSON_CLS:
                persons.append((x1, y1, x2, y2))
            elif cls_id in MACHINERY_CLASSES:
                mcx, mcy = (x1 + x2) // 2, y2
                is_ignored = False
                
                if x2 > width * 0.65 and y2 > height * 0.55:
                    is_ignored = True
                elif x1 < width * 0.45 and y2 > height * 0.60:
                    is_ignored = True

                if y1 < height * 0.50:
                    is_ignored = False

                if is_ignored:
                    ignored_machineries.append((x1, y1, x2, y2, mcx, mcy))
                else:
                    speed_kmh = 0.0
                    if obj_id is not None:
                        if obj_id in track_history:
                            prev_x, prev_y, prev_frame, prev_speed = track_history[obj_id]
                            
                            dist_px = math.hypot(mcx - prev_x, mcy - prev_y)
                            time_s = (frame_count - prev_frame) / fps
                            
                            if time_s > 0:
                                if dist_px < MIN_PIXEL_MOVE:
                                    current_speed_kmh = 0.0
                                else:
                                    dist_m = dist_px * PIXEL_TO_METER
                                    current_speed_kmh = (dist_m / time_s) * 3.6
                                
                                speed_kmh = (current_speed_kmh * 0.6) + (prev_speed * 0.4)
                                
                                if speed_kmh < SPEED_THRESHOLD_KMH:
                                    speed_kmh = 0.0
                            else:
                                speed_kmh = prev_speed
                        
                        track_history[obj_id] = (mcx, mcy, frame_count, speed_kmh)
                    
                    valid_machineries.append((x1, y1, x2, y2, mcx, mcy, speed_kmh, obj_id))

        # [수정] 다각형 오버레이 채우기 (반투명 주황색)
        for x1, y1, x2, y2, mcx, mcy, speed_kmh, obj_id in valid_machineries:
            poly = WORK_POLYGON + [mcx, mcy]
            cv2.fillPoly(overlay, [poly], (0, 165, 255))

        for x1, y1, x2, y2, mcx, mcy in ignored_machineries:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), 2)
            cv2.putText(overlay, "PARKED", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)

        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        intrusion_count = 0
        no_helmet_count = 0

        # [수정] 다각형 윤곽선 그리기
        for x1, y1, x2, y2, mcx, mcy, speed_kmh, obj_id in valid_machineries:
            poly = WORK_POLYGON + [mcx, mcy]
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 140, 255), thickness=2)
            
            # 텍스트 위치 조절 (상단 끝에서 살짝 띄움)
            cv2.putText(frame, "EQUIPMENT ZONE", (mcx - 60, mcy - 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            display_speed = "0" if speed_kmh == 0.0 else f"{speed_kmh:.1f}"
            cv2.putText(frame, f"SPEED: {display_speed} km/h", (mcx - 60, mcy - 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for px1, py1, px2, py2 in persons:
            pcx, pcy = (px1 + px2) // 2, py2 
            warning_msgs = []
            box_color = (0, 255, 0) 

            # [수정] 다각형 내부 침범 판별 (cv2.pointPolygonTest 사용)
            for x1, y1, x2, y2, mcx, mcy, speed_kmh, obj_id in valid_machineries:
                poly = WORK_POLYGON + [mcx, mcy]
                # >= 0 이면 점이 다각형 내부에 있거나 경계선에 걸쳐있음을 의미함
                if cv2.pointPolygonTest(poly, (pcx, pcy), False) >= 0:
                    warning_msgs.append("RADIUS INTRUSION")
                    box_color = (0, 0, 255) 
                    intrusion_count += 1 
                    cv2.line(frame, (pcx, pcy), (mcx, mcy), (0, 0, 255), 2)

            has_helmet = check_helmet(frame, px1, py1, px2, py2)
            if has_helmet:
                helmet_text = "HELMET: OK"
            else:
                helmet_text = "NO HELMET"
                no_helmet_count += 1 
                if box_color == (0, 255, 0): box_color = (0, 165, 255) 

            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(frame, helmet_text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            for i, msg in enumerate(warning_msgs):
                cv2.putText(frame, msg, (px1, py1 - 25 - (i * 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        video_time_sec = frame_count / fps
        m, s = divmod(int(video_time_sec), 60)
        video_time_str = f"{m:02d}:{s:02d}"

        if intrusion_count > 0:
            status_text = "DANGER"
            status_color = (0, 0, 255) 
            if video_time_sec - last_log_time["DANGER"] > LOG_COOLDOWN_SEC:
                msg = f"INTRUSION ({intrusion_count}p)"
                write_log("DANGER", msg, video_time_str)
                last_log_time["DANGER"] = video_time_sec
                recent_events.append((video_time_str, "DANGER", msg))

        elif no_helmet_count > 0:
            status_text = "WARNING"
            status_color = (0, 165, 255) 
            if video_time_sec - last_log_time["WARNING"] > LOG_COOLDOWN_SEC:
                msg = f"NO HELMET ({no_helmet_count}p)"
                write_log("WARNING", msg, video_time_str)
                last_log_time["WARNING"] = video_time_sec
                recent_events.append((video_time_str, "WARNING", msg))
        else:
            status_text = "SAFE"
            status_color = (0, 255, 0) 

        if len(recent_events) > MAX_EVENTS_DISPLAY:
            recent_events.pop(0)

        # ==========================================
        # 대시보드 렌더링
        # ==========================================
        dash_x, dash_y = 20, 20
        dash_w = 320 
        event_display_height = len(recent_events) * 20 if recent_events else 0
        dash_h = 130 + (len(valid_machineries) * 30) + event_display_height + (30 if recent_events else 0)
        
        dash_overlay = frame.copy()
        cv2.rectangle(dash_overlay, (dash_x, dash_y), (dash_x + dash_w, dash_y + dash_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(dash_overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, f"STATUS: {status_text}", (dash_x + 15, dash_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(frame, f"Intrusion Count: {intrusion_count}", (dash_x + 15, dash_y + 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"No Helmet Count: {no_helmet_count}", (dash_x + 15, dash_y + 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "-" * 28, (dash_x + 15, dash_y + 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        y_offset = dash_y + 140
        for x1, y1, x2, y2, mcx, mcy, speed_kmh, obj_id in valid_machineries:
            display_speed = "0" if speed_kmh == 0.0 else f"{speed_kmh:.1f}"
            eq_name = f"EQ-{obj_id}" if obj_id is not None else "EQ-X"
            cv2.putText(frame, f"{eq_name} Speed: {display_speed} km/h", (dash_x + 15, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 30

        if recent_events:
            cv2.putText(frame, "-" * 28, (dash_x + 15, y_offset - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "RECENT EVENTS:", (dash_x + 15, y_offset + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 30
            
            for t_str, e_type, msg in recent_events:
                txt_color = (0, 0, 255) if e_type == "DANGER" else (0, 165, 255)
                cv2.putText(frame, f"[{t_str}] {e_type}: {msg}", (dash_x + 15, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_color, 1)
                y_offset += 20

        cv2.imshow("Construction Safety Monitor", frame)

        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()