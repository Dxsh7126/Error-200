from ultralytics import YOLO
import cv2
import os

#pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  


image_folder = r"C:\Users\Ashwini\Desktop\Disaster AI in social media picture analysis"

# function to simulate GPS + emergency response
def send_to_emergency_services(file_path, coords=(12.9716, 77.5946)): #Assuming coordinates
    print(f"\nALERT: Damage detected in {file_path}")
    print(f"Sending GPS coordinates: {coords}")
    print("Emergency services contacted!\n")

# Function to process video 
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)

        # If damage is detected
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            send_to_emergency_services(video_path)
            break

    cap.release()
    print(f"Finished scanning video: {video_path}\n")

# Loop through files and process
for file_name in os.listdir(image_folder):
    file_path = os.path.join(image_folder, file_name)

    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"Scanning image: {file_name}...")

        results = model(file_path)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            send_to_emergency_services(file_path)
        else:
            print("No damage detected in image. Moving to next.\n")

    elif file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print(f"Scanning video: {file_name}...")
        process_video(file_path)

    else:
        print(f"Skipping unsupported file type: {file_name}")
