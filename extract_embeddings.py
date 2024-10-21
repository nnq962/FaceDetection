import face_recognition
import cv2

class FaceEmbeddingExtractor:
    def __init__(self):
        self.embeddings = []

    def extract_embedding(self, frame):
        """
        Lấy embedding cho một khuôn mặt từ một bức ảnh.
        :param frame: Khung hình chứa khuôn mặt (dạng BGR - OpenCV).
        :return: Vector embedding của khuôn mặt hoặc None nếu không tìm thấy.
        """
        # Chuyển frame từ BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Tìm vị trí của khuôn mặt (sử dụng face_recognition)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) > 0:
            # Lấy embedding cho khuôn mặt đầu tiên
            embedding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            return embedding
        else:
            return None

    def extract_embeddings(self, frame, face_locations):
        """
        Lấy embedding cho nhiều khuôn mặt từ một khung hình, với thông tin về vị trí khuôn mặt đã có sẵn.
        :param frame: Khung hình chứa khuôn mặt (dạng BGR - OpenCV).
        :param face_locations: Danh sách các bounding boxes của khuôn mặt trong khung hình.
        :return: Danh sách các vector embedding cho từng khuôn mặt.
        """
        # Chuyển frame từ BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Lấy embedding cho tất cả các khuôn mặt dựa vào face_locations
        embeddings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        return embeddings
