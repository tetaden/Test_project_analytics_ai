import cv2
from ultralytics import YOLO
# Загружаем нашу модельку
model = YOLO("data/best.pt")

# Открываем веб-камеру
cap = cv2.VideoCapture(0) 

# Проверяем, что все ок (в Goole Colab не работало)
if not cap.isOpened():
    print("Ошибка: не удалось открыть веб-камеру")
    exit()

while True:
    # Читаем кадр с веб-камеры
    ret, frame = cap.read()
    if not ret:
        break

    # инференс модели
    results = model(frame)

    # Находим Bounding Box
    annotated_frame = results[0].plot()

    # Отображапем кадр и bounding box
    cv2.imshow("Finding mugs", annotated_frame)

    # Чтобы выйти надо нажать enter
    if cv2.waitKey(1) == 13:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()