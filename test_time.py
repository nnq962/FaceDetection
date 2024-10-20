import cv2
from deepface import DeepFace
import time
import tensorflow as tf

# Tắt GPU
tf.config.set_visible_devices([], 'GPU')

# Kiểm tra lại các thiết bị hiện có
print("Devices available:", tf.config.list_physical_devices())


# Đọc ảnh từ file
image_path = "manypeople.jpeg"
image = cv2.imread(image_path)

# Ghi lại thời gian bắt đầu
start_time = time.time()

# Sử dụng extract_faces để phát hiện và trích xuất tất cả các khuôn mặt
faces = DeepFace.extract_faces(image, enforce_detection=False, detector_backend="yolov8")

print(len(faces))

# Ghi lại thời gian kết thúc
end_time = time.time()

# Tính toán thời gian chạy
execution_time = end_time - start_time
print(f"Thời gian chạy: {execution_time:.2f} giây")

# Duyệt qua các khuôn mặt được phát hiện và vẽ hình chữ nhật
for face in faces:
    # In ra thông tin về khuôn mặt để kiểm tra

    # Lấy thông tin về khuôn mặt
    facial_area = face['facial_area']  # Tọa độ của khuôn mặt
    x, y = facial_area['x'], facial_area['y']
    w, h = facial_area['w'], facial_area['h']

    # Vẽ hình chữ nhật quanh khuôn mặt
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Màu xanh lá cây, độ dày 2



# Hiển thị ảnh với các hình chữ nhật
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
