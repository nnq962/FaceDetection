import cv2

# Tải mô hình phát hiện khuôn mặt Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mở webcam
cap = cv2.VideoCapture(0)  # Thay đổi thành 1 để sử dụng webcam thứ hai

if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận diện khung hình.")
        break

    # Chuyển đổi khung hình sang ảnh xám để phát hiện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Vẽ hình chữ nhật quanh khuôn mặt phát hiện được
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Lật khung hình theo chiều ngang
    flipped_frame = cv2.flip(frame, 1)

    # Hiển thị khung hình lật
    cv2.imshow('Webcam - Flipped', flipped_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng tất cả các cửa sổ
cap.release()
cv2.destroyAllWindows()
