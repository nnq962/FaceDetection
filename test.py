import extract_embeddings
import cv2
import face_recognition
import time
import face_detection

image_path = "photo_test/nnq_test.png"

image = cv2.imread(image_path)

fd = face_detection.SSDFaceDetectorOpenCV(threshold=0.8)
get_embeddings = extract_embeddings.FaceEmbeddingExtractor()

faces = fd.detect_faces(image)

result = get_embeddings.extract_embeddings(image, faces)
result2 = get_embeddings.extract_embedding(image)
print("so luong khuon mat: ", len(faces))

print(result)
print(len(result))

print("======================")

print(result2)
print(len(result2))



