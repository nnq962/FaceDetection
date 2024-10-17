import os
import json
import csv

# Đường dẫn tới thư mục lưu các file embedding
embedding_dir = 'embeddings'
csv_file = 'metadata.csv'  # File CSV để lưu metadata

# Kiểm tra xem file CSV đã tồn tại hay chưa và đọc dữ liệu hiện có
existing_data = set()
if os.path.exists(csv_file):
    with open(csv_file, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Bỏ qua tiêu đề
        for row in reader:
            existing_data.add(row[2])  # Thêm đường dẫn vào set

# Mở file CSV để ghi (append mode)
with open(csv_file, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Duyệt qua tất cả các file JSON trong thư mục embedding
    for idx, filename in enumerate(os.listdir(embedding_dir)):
        if filename.endswith('.json'):
            # Đường dẫn tới file JSON
            path = os.path.join(embedding_dir, filename)

            # Kiểm tra nếu đường dẫn đã tồn tại trong dữ liệu hiện có
            if path in existing_data:
                print(f"Đường dẫn {path} đã tồn tại. Bỏ qua.")
                continue  # Bỏ qua nếu đã tồn tại

            # Tách tên từ tên file (ví dụ: name_embedding_index.json)
            parts = filename.split('_')
            name = parts[0]

            # Ghi vào file CSV
            writer.writerow([name, idx, path])
            print(f"Đã thêm: {name}, {idx}, {path}")

print(f"Metadata đã được lưu vào file {csv_file}")
