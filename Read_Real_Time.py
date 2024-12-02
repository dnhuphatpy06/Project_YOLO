import cv2
from ultralytics import YOLO
import random

model = YOLO("yolov8n.pt")
cam_index = 0
classes = []


def generate_random_color():
    """Tạo một màu ngẫu nhiên dưới dạng (B, G, R)"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def Read_RealTime(cam_index, selected_classes=None):
    """
    Nhận diện đối tượng trong thời gian thực từ webcam.

    Args:
        cam_index (int): Chỉ số webcam (0, 1, 2,...).
        selected_classes (list): Danh sách các lớp đối tượng cần nhận diện. Nếu là None, nhận diện tất cả các lớp.

    Returns:
        None: Hàm sẽ hiển thị video trực tiếp.
    """
    if selected_classes is None or len(selected_classes) == 0:
        selected_classes = ['all']
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Không thể mở webcam.")
        return
    class_colors = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        detected_classes = results[0].boxes.cls  
        boxes = results[0].boxes.xyxy  
        confidences = results[0].boxes.conf  
        class_names = results[0].names  
        for box, cls, conf in zip(boxes, detected_classes, confidences):
            class_name = class_names[int(cls)]  
            if class_name in selected_classes or 'all' in selected_classes:
                x1, y1, x2, y2 = map(int, box[:4])  
                if class_name not in class_colors:
                    class_colors[class_name] = generate_random_color()
                color = class_colors[class_name]  
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{class_name} {conf:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_width, text_height = text_size
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_width, y1), color, -1)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("YOLO Object Detection Real Time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
Read_RealTime(cam_index, classes)