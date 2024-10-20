import cv2
import numpy as np

prototxt_path = "ssd_model/deploy.prototxt"
model_path = "ssd_model/res10_300x300_ssd_iter_140000.caffemodel"

class SSDFaceDetectorOpenCV:
    def __init__(self, threshold=0.65):
        """
        Khởi tạo SSD face detector.
        :param threshold: Ngưỡng để quyết định khuôn mặt (mặc định là 0.5).
        """
        self.prototxt = prototxt_path
        self.model = model_path
        self.detector = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        self.threshold = threshold

    def detect_faces(self, frame):
        """
        Phát hiện khuôn mặt trong một frame.
        :param frame: Khung hình từ camera.
        :return: Danh sách tọa độ các bounding box của khuôn mặt.
        """
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.detector.setInput(blob)
        detections = self.detector.forward()

        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype(int))

        return faces