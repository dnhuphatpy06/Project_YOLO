import cv2
from ultralytics import YOLO
import os
import random

model = YOLO("yolov8n.pt")
img_path = os.path.dirname(os.path.abspath(__file__)) + "/anh_test.jpg"
classes = []

def generate_random_color():
    """Tạo một màu ngẫu nhiên dưới dạng (B, G, R)"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def Read_Img(image_path, selected_classes):
    """
    Nhận diện đối tượng trong ảnh bằng YOLO và vẽ bounding box lên các đối tượng được chọn.
    Vẽ một khung chữ nhật ở trên để hiển thị tên lớp và confidence.

    Args:
        image_path (str): Đường dẫn đến file ảnh cần nhận dạng.
        selected_classes (list): Danh sách các lớp đối tượng cần nhận dạng. Nếu danh sách rỗng, nhận diện tất cả.

    Returns:
        None: Hàm hiển thị ảnh đã nhận diện.
    """
    if not selected_classes or len(selected_classes) == 0:
        selected_classes = ['all']
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    detected_classes = results[0].boxes.cls
    boxes = results[0].boxes.xyxy
    confidences = results[0].boxes.conf
    class_names = results[0].names
    color_map = {}
    for box, cls, conf in zip(boxes, detected_classes, confidences):
        class_name = class_names[int(cls)]
        if class_name in selected_classes or 'all' in selected_classes:
            if class_name not in color_map:
                color_map[class_name] = generate_random_color()
            x1, y1, x2, y2 = map(int, box[:4]) 
            color = color_map[class_name]  
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{class_name} {conf:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_width, text_height = text_size
            cv2.rectangle(img, (x1, y1 - 30), (x1 + text_width, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow("Detected Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
Read_Img(img_path, classes)
