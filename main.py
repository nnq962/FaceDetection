import camera
import cv2

cam_manager = camera.CameraManager()

while True:
    # Lấy frame từ camera
    frame = cam_manager.get_frame()

    # Phát hiện khuôn mặt
    boxes = cam_manager.detect_faces(frame)

    # Vẽ các bounding box khuôn mặt
    frame_with_faces = cam_manager.draw_faces(frame, boxes)

    # Hiển thị frame
    cv2.imshow('Camera with Face Detection', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Giải phóng camera
cam_manager.release()