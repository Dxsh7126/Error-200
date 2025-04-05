import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Drone camera feed (change to drone input if needed)
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

# Log file
log_file = open("disaster_report.txt", "a")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 640)) #Resizing to smaller for more frame rate
    output = frame.copy()
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect humans
    results = model(frame, verbose=False)[0]
    person_box = []

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        if cls_id == 0 and conf > 0.3:
            x1, y1, x2, y2 = map(int, box.xyxy[0])      #Boxing the person
            person_box.append((x1, y1, x2, y2))
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, f"Person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1) #Puts text on the box


    #FLOOD DETECTION
    min_flood = np.array([10, 40, 40])     # Muddy water color is 10 40 40 in rgb
    max_flood = np.array([30, 255, 255])
    flood_mask = cv2.inRange(hsv, min_flood, max_flood)

    movement_mask = fgbg.apply(blurred)         #Motion detection
    movement_mask = cv2.threshold(movement_mask, 200, 255, cv2.THRESH_BINARY)[1]        #Gives clear motion zones
    flood_mask = cv2.bitwise_and(flood_mask, cv2.bitwise_not(movement_mask))

    contours, _ = cv2.findContours(flood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(cont)
            box = (x, y, x+w, y+h)
            overlap = any(                                                          #Overlapping humans so that it doesnt mistake a human for earthquake
                not (box[2] < px1 or box[0] > px2 or box[3] < py1 or box[1] > py2)
                for px1, py1, px2, py2 in person_box
            )
            if not overlap:
                cv2.rectangle(output, (x, y), (x+w, y+h), (165, 42, 42), 2)
                cv2.putText(output, "Flood", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 75, 0), 2)
                log_file.write(f"[{datetime.now()}] Flood detected at x={x}, y={y}, w={w}, h={h}\n")

    #FIRE DETECTION
    min_fire = np.array([0, 100, 100]) #Fire color is 0 100 100 in RGB
    max_fire = np.array([25, 255, 255])
    fire_mask = cv2.inRange(hsv, min_fire, max_fire)
    bright_mask = cv2.inRange(hsv[:, :, 2], 180, 255)
    fire_mask = cv2.bitwise_and(fire_mask, bright_mask)

    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 200:
            x, y, w, h = cv2.boundingRect(cont)
            box = (x, y, x+w, y+h)
            overlap = any(                                                          #Overlapping humans so that it doesnt mistake a human for earthquake
                not (box[2] < px1 or box[0] > px2 or box[3] < py1 or box[1] > py2)
                for px1, py1, px2, py2 in person_box
            )
            if not overlap:
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(output, "Fire", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                log_file.write(f"[{datetime.now()}] Fire detected at x={x}, y={y}, w={w}, h={h}\n")

    #EARTHQUAKE DAMAGE DETECTION
    blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred_gray, 80, 180)
    edges_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Finds blobs
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 3000:
            x, y, w, h = cv2.boundingRect(cont)         # Boxing the blobs
            box = (x, y, x+w, y+h)
            overlap = any(                                                          #Overlapping humans so that it doesnt mistake a human for earthquake
                not (box[2] < px1 or box[0] > px2 or box[3] < py1 or box[1] > py2)
                for px1, py1, px2, py2 in person_box
            )
            if not overlap:
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(output, "Earthquake Damage", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                log_file.write(f"[{datetime.now()}] Earthquake Damage at x={x}, y={y}, w={w}, h={h}\n")         #File handling

    cv2.imshow("Drone Disaster Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('e'):       #To exit the camera
        break

log_file.close()
cap.release()
cv2.destroyAllWindows()
