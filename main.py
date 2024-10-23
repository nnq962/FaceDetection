import camera
import extract_embeddings
import cv2
import time
import verify_frame
import numpy as np

embedding_folder = 'embeddings_data'
cam_manager = camera.CameraManager(camera_id=0, detector="mediapipe")
verifier = verify_frame.VerifyFrame(embedding_folder)

# Lưu thời điểm bắt đầu
start_time = time.time()

result = []
boxes = []

while True:
    # Lấy frame từ camera
    frame = cam_manager.get_frame(flip_code=1)

    boxes = cam_manager.detect_faces(frame)

    current_time = time.time()
    # Nếu đã qua 1 giây kể từ lần lấy embedding trước
    if current_time - start_time >= 1:
        # Lấy embedding của các khuôn mặt sau mỗi giây
        embeddings = cam_manager.get_embs(frame, boxes)
        result = verifier.verify(embeddings)

        # Cập nhật thời gian bắt đầu cho lần sau
        start_time = current_time

    # Vẽ các bounding box khuôn mặt
    frame_with_faces = cam_manager.draw_faces(frame, boxes, verifier.get_name())

    # Hiển thị frame
    cv2.imshow('Camera with Face Detection', frame_with_faces)  

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera
cam_manager.release()
