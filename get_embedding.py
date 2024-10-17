import os
import json
from deepface import DeepFace

# Thư mục chứa ảnh
root_dir = 'photos'

# Thư mục chứa embedding
embedding_dir = 'embeddings'
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)

# Duyệt qua từng thư mục (mỗi thư mục là một người)
for person_name in os.listdir(root_dir):
    person_dir = os.path.join(root_dir, person_name)
    
    # Duyệt qua từng ảnh trong thư mục
    for i, image_name in enumerate(os.listdir(person_dir)):
        image_path = os.path.join(person_dir, image_name)
        
        # Lấy embedding
        embedding = DeepFace.represent(img_path=image_path, model_name='Facenet')

        # Lưu embedding ra file JSON
        file_name = f"{person_name}_embedding_{i}.json"
        file_path = os.path.join(embedding_dir, file_name)
        
        with open(file_path, 'w') as f:
            json.dump(embedding, f)

"""
lay embedding cho toan bo anh

import os
import json
from deepface import DeepFace

# Thư mục chứa ảnh
root_dir = 'photos'

# Thư mục chứa embedding
embedding_dir = 'embeddings'
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)

# Duyệt qua toàn bộ ảnh mà không cần phân loại trước
for i, image_name in enumerate(os.listdir(root_dir)):
    image_path = os.path.join(root_dir, image_name)
    
    # Lấy embedding
    embedding = DeepFace.represent(img_path=image_path, model_name='Facenet')

    # Lấy tên người từ tên file (giả sử ảnh được đặt tên theo người, ví dụ: john_doe_01.jpg)
    person_name = image_name.split('_')[0]
    
    # Lưu embedding ra file JSON
    file_name = f"{person_name}_embedding_{i}.json"
    file_path = os.path.join(embedding_dir, file_name)
    
    with open(file_path, 'w') as f:
        json.dump(embedding, f)
"""