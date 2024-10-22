import extract_embeddings
import matplotlib.pyplot as plt
import face_detection
import cv2

get_embs = extract_embeddings.FaceEmbeddingExtractor()
image_path = "photo_test/nnq1.jpg"
image = cv2.imread(image_path)
ssd = face_detection.SSDFaceDetectorOpenCV()

faces = ssd.detect_faces(image)
get_embs.extract_embeddings(image, faces)