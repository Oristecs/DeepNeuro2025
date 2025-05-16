# Импортируем библиотеки

from ultralytics import YOLO

model = YOLO("yolov8s.pt")
# обучаем на собственном датасете
results = model.train(data="dogs.yaml", model="yolov8s.pt", epochs=1, batch=8,
                      project='dogs', val = True, verbose=True)
# после обучение ядро перезапускается, 
# поэтому заного импортируем библиотеки и инициализируем переменные


from ultralytics import YOLO
import cv2


model = YOLO("yolov8s.pt")

results = model("dogs0.JPG")

# результат
result = results[0]
cv2.imshow("YOLOv8", result.plot())
