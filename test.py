from deepface import DeepFace
import cv2
import time

image_path = "photo_test/nnq_test.png"
image = cv2.imread(image_path)

start_time = time.time()
result = DeepFace.represent(img_path=image_path)
end_time = time.time()
print(result, end_time - start_time)