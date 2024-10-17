from deepface import DeepFace

result = DeepFace.verify(
  img1_path = "anhthe.jpg",
  img2_path = "data_face_nnq/face_0.jpg",
)


print(result)