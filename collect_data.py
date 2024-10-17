import cv2
import os

# Tạo thư mục để lưu ảnh nếu chưa có
save_dir = "data_face_nnq"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Khởi động webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Nhận diện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật bao quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Lưu khuôn mặt
        face = gray[y:y + h, x:x + w]
        file_name = os.path.join(save_dir, f"face_{count}.jpg")
        cv2.imwrite(file_name, face)
        count += 1

    # Hiển thị video với khuôn mặt được nhận diện
    cv2.imshow('Collecting Faces', frame)

    # Nhấn 'q' để dừng
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 200:  # Lưu 100 ảnh
        break

cap.release()
cv2.destroyAllWindows()
