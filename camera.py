import cv2
import extract_embeddings
import ssd_detect
import mtcnn_detect
import mediapipe_detect

class CameraManager:
    def __init__(self, camera_id=0, frame_width=640, frame_height=480, detector=None):
        """
        Khởi tạo camera manager với các thông số camera.
        :param camera_id: ID của camera (mặc định là 0).
        :param frame_width: Chiều rộng khung hình (mặc định 640).
        :param frame_height: Chiều cao khung hình (mặc định 480).
        """
        self.cap = cv2.VideoCapture(camera_id) 
        if not self.cap.isOpened():
            raise ValueError("Cannot open camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        
        self.get_embeddings = extract_embeddings.FaceEmbeddingExtractor()
        
        # Xác định detector
        if detector is None or 'ssd':
            self.face_detection = ssd_detect.SSDFaceDetectorOpenCV()
        elif detector == 'mediapipe':
            self.face_detection = mediapipe_detect.MediapipeDetector()
        elif detector == 'mtcnn':
            self.face_detection = mtcnn_detect.MTCNNFaceDetector()
        else:
            raise ValueError("Unknown detector type")
        
        print("================", detector, "================")

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
    def draw_faces(frame, boxes, names):
        """
        Vẽ các bounding box khuôn mặt lên frame và thêm tên người.
        :param frame: Frame từ camera.
        :param boxes: Danh sách tọa độ các bounding box khuôn mặt.
        :param names: Danh sách tên tương ứng với các khuôn mặt.
        :return: Frame với bounding box khuôn mặt và tên người.
        """

        if boxes is None or len(boxes) == 0:
            return frame  # Nếu không có khuôn mặt nào thì trả về frame ban đầu

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]

            # Vẽ bounding box quanh khuôn mặt
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Gắn tên lên bounding box
            name = names[i] if i < len(names) else "Unknown"
            font_scale = 0.7
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
        if len(face_locations) == 0:
            return []
        return self.get_embeddings.extract_embeddings(frame, face_locations)