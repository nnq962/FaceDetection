import cv2

import face_detection

class CameraManager:
    def __init__(self, camera_id=0, frame_width=640, frame_height=480):
        """
        Khởi tạo camera manager với các thông số camera.
        :param camera_id: ID của camera (mặc định là 0).
        :param frame_width: Chiều rộng khung hình (mặc định 640).
        :param frame_height: Chiều cao khung hình (mặc định 480).
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.face_detection = face_detection.SSDFaceDetectorOpenCV(0.65)

    def get_frame(self):
        """
        Lấy một frame từ camera.
        :return: Frame lấy từ camera (ảnh màu).
        """
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Không thể đọc từ camera.")
        return frame

    def detect_faces(self, frame):
        """
        Phát hiện khuôn mặt trong một frame bằng MTCNN.
        :param frame: Frame từ camera.
        :return: Danh sách tọa độ các bounding box của khuôn mặt.
        """
        return self.face_detection.detect_faces(frame)

    @staticmethod
    def draw_faces(frame, boxes):
        """
        Vẽ các bounding box khuôn mặt lên frame.
        :param frame: Frame từ camera.
        :param boxes: Danh sách tọa độ các bounding box khuôn mặt.
        :return: Frame với bounding box khuôn mặt.
        """
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def release(self):
        """
        Giải phóng camera.
        """
        self.cap.release()
        cv2.destroyAllWindows()

