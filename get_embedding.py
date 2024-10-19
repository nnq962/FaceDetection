import os
import json
import cv2
import insightface

# Thư mục chứa ảnh
root_dir = 'database'

# Thư mục chứa embedding
embedding_dir = 'embeddings'
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)

# Khởi tạo mô hình InsightFace
model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
model.prepare(ctx_id=0)

# Duyệt qua từng thư mục (mỗi thư mục là một người)
for person_name in os.listdir(root_dir):
    person_dir = os.path.join(root_dir, person_name)

    # Danh sách để lưu tất cả embeddings cho mỗi người
    embeddings_list = []

    # Duyệt qua từng ảnh trong thư mục
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)

        # Đọc ảnh sử dụng OpenCV
        image = cv2.imread(image_path)
        faces = model.get(image)

        if len(faces) > 0:  # Nếu phát hiện khuôn mặt và có embedding
            embedding = faces[0].embedding.tolist()  # Lấy embedding và chuyển sang list để lưu JSON

            # Thêm embedding và tên hình ảnh vào danh sách
            embeddings_list.append({
                "image_name": image_name,
                "embedding": embedding
            })
        else:
            print(f"Không tìm thấy khuôn mặt trong ảnh {image_name}")

    # Sau khi xử lý xong tất cả các ảnh của một người, lưu danh sách embeddings vào file JSON
    file_name = f"{person_name}_embeddings.json"
    file_path = os.path.join(embedding_dir, file_name)

    with open(file_path, 'w') as f:
        json.dump({"name": person_name, "embeddings": embeddings_list}, f)

    print(f"Đã lưu embeddings cho {person_name} vào {file_name}")
