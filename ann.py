import faiss
import numpy as np

np.random.seed(1)

# Giả sử bạn có 100 embedding cho các hình ảnh trong database
embeddings = np.random.random((100, 512)).astype('float32')  # (100, 512) 100 embedding với 512 chiều

# Tạo một embedding mới cần kiểm tra
test_embedding = np.random.random((1, 512)).astype('float32')


# Tạo chỉ mục FAISS với khoảng cách L2 (Euclidean distance)
dimension = 512  # Kích thước của vector embedding
index = faiss.IndexFlatL2(dimension)  # Chỉ mục sử dụng L2 cho tìm kiếm

# Thêm các embedding vào chỉ mục FAISS
index.add(embeddings)


# Tìm kiếm 5 embedding gần nhất với embedding kiểm tra
k = 5  # Số lượng embedding gần nhất cần tìm
D, I = index.search(test_embedding, k)  # D là khoảng cách, I là chỉ số của các embedding gần nhất

# In kết quả
print(f"Khoảng cách: {D}")
print(f"Chỉ số của các embedding gần nhất: {I}")


nearest_embedding = embeddings[I[0][0]]

nlist = 10  # Số lượng phân vùng (với dữ liệu nhỏ bạn có thể chọn giá trị thấp)
quantizer = faiss.IndexFlatL2(dimension)  # Chỉ mục L2 cho việc lượng tử hóa
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)  # Tạo chỉ mục IVF

# Đào tạo chỉ mục (cần thiết với IndexIVFFlat)
index.train(embeddings)

# Thêm embedding vào chỉ mục
index.add(embeddings)

# Tìm kiếm embedding gần nhất với ANN
D, I = index.search(test_embedding, k)

print(f"Khoảng cách ANN: {D}")
print(f"Chỉ số của các embedding gần nhất ANN: {I}")