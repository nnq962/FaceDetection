import cv2
import extract_embeddings
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
        self.face_detection = face_detection.SSDFaceDetectorOpenCV(0.5)
        self.get_embeddings = extract_embeddings.FaceEmbeddingExtractor()
        self.embeddings = None

    def get_frame(self, flip_code=None):
        """
        Lấy một frame từ camera và có thể lật frame nếu cần.
        :param flip_code: Mã lật (0: lật dọc, 1: lật ngang, -1: lật cả hai).
        :return: Frame lấy từ camera (ảnh màu), đã lật nếu được chỉ định.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Không thể đọc từ camera.")
        
        if flip_code is not None:
            frame = cv2.flip(frame, flip_code)

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
        Vẽ các bounding box khuôn mặt lên frame và thêm tên người.
        :param frame: Frame từ camera.
        :param boxes: Danh sách tọa độ các bounding box khuôn mặt.
        :return: Frame với bounding box khuôn mặt và tên người.
        """
        for box in boxes:
            y1, x2, y2, x1 = [int(b) for b in box]
            
            # Vẽ bounding box quanh khuôn mặt
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Gắn tên tạm thời "Test Name" lên bounding box
            name = "Test Name"
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (x1, y1 - 10), font, font_scale, (0, 0, 255), 1)
        
        return frame

    def release(self):
        """
        Giải phóng camera.
        """
        self.cap.release()
        cv2.destroyAllWindows()

    def get_embs(self, frame, face_locations):
        return self.get_embeddings.extract_embeddings(frame, face_locations)