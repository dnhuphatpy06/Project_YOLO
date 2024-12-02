import cv2
from ultralytics import YOLO
import random
import os

model = YOLO("yolov8n.pt")  
video_path = os.path.dirname(os.path.abspath(__file__)) + "/video_test.mp4"
classes = []

def generate_random_color():
    """Tạo màu ngẫu nhiên dưới dạng (B, G, R)"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def Read_Video(video_path, selected_classes=None):
    if not selected_classes or len(selected_classes) == 0:
        selected_classes = ['all']
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video.")
        return
    class_colors = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        detected_classes = results[0].boxes.cls  
        boxes = results[0].boxes.xyxy  
        confidences = results[0].boxes.conf  
        class_names = results[0].names  
        for box, cls, conf in zip(boxes, detected_classes, confidences):
            class_name = class_names[int(cls)] 
            if class_name not in class_colors:
                class_colors[class_name] = generate_random_color()
            color = class_colors[class_name] 
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = f"{class_name} {conf:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]  
            text_width, text_height = text_size
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  
        resized_frame = cv2.resize(frame, (800, 600))
        cv2.imshow("Detected Video", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

Read_Video(video_path, classes)
