import os
import json
import cv2
import face_detection
import extract_embeddings
import numpy as np

# Khởi tạo các đối tượng cần thiết
get_embs = extract_embeddings.FaceEmbeddingExtractor()
ssd = face_detection.SSDFaceDetectorOpenCV()

# Đường dẫn thư mục
database_path = "database"
output_path = "embeddings"

# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists(output_path):
    os.makedirs(output_path)


def extract_embeddings_from_image(image_path):
    image = cv2.imread(image_path)
    # Phát hiện vị trí khuôn mặt
    face_locations = ssd.detect_faces(image)
    if len(face_locations):
        return get_embs.extract_embeddings(image, face_locations)
    else:
        print(f"No face detected in {image_path}")
        return []


def convert_ndarray_to_list(ndarrays):
    """
    Chuyển đổi các ndarray thành list để có thể lưu dưới dạng JSON.
    :param ndarrays: Danh sách các ndarray.
    :return: Danh sách các list đã chuyển đổi.
    """
    return [ndarray.tolist() for ndarray in ndarrays]


def process_database():
    for person_name in os.listdir(database_path):
        person_dir = os.path.join(database_path, person_name)

        # Kiểm tra xem thư mục con có tồn tại không (chứa ảnh)
        if os.path.isdir(person_dir):
            embeddings_list = []

            # Duyệt qua tất cả các hình ảnh trong thư mục của mỗi người
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)

                # Trích xuất embeddings
                embeddings = extract_embeddings_from_image(image_path)

                if embeddings:
                    # Chuyển đổi từ ndarray sang list để lưu JSON
                    embeddings_list.extend(convert_ndarray_to_list(embeddings))

            # Lưu các embeddings vào file JSON
            if embeddings_list:
                output_file = os.path.join(output_path, f"{person_name}.json")
                data = {
                    "name": person_name,
                    "embeddings": embeddings_list
                }
                with open(output_file, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                print(f"Saved embeddings for {person_name} to {output_file}")
            else:
                print(f"No embeddings found for {person_name}")


if __name__ == "__main__":
    process_database()
