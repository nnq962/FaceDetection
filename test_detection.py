import extract_embeddings
import face_detection
import face_recognition
import cv2

obj1 = face_detection.SSDFaceDetectorOpenCV()
obj2 = extract_embeddings.FaceEmbeddingExtractor()

image_path = "photo_test/putin.jpg"
image = cv2.imread(image_path)

faces = obj1.detect_faces(image)
kq1 = obj2.extract_embedding(image)
kq2 = obj2.extract_embeddings(image, faces)

print(kq1)
print("================")
print(kq2)


result = face_recognition.compare_faces(kq2, kq1)
print(result)