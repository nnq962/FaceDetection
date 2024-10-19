import insightface
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load mô hình ArcFace torch (buffalo_l)
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0)  # ctx_id=-1 nếu không có GPU

# Đọc ảnh bằng OpenCV
img1_path = 'nnq1.jpg'
img2_path = 'nnq2.jpg'

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Chuyển ảnh từ BGR sang RGB vì InsightFace yêu cầu ảnh phải là RGB
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


# Lấy các khuôn mặt từ cả hai ảnh
faces_img1 = model.get(img1_rgb)
faces_img2 = model.get(img2_rgb)

print("------------------")
print(len(faces_img1[0].embedding))

# Kiểm tra và lấy embedding
if len(faces_img1) > 0 and len(faces_img2) > 0:
    embedding1 = faces_img1[0].embedding
    embedding2 = faces_img2[0].embedding

    # So sánh cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    print(f"Cosine Similarity: {similarity}")

    threshold = 0.5
    if similarity > threshold:
        print("Hai ảnh là của cùng một người.")
    else:
        print("Hai ảnh không phải của cùng một người.")
else:
    print("Không tìm thấy khuôn mặt trong một trong hai ảnh.")
