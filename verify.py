import face_recognition
import json
import os


class VerifyFrame:
    def __init__(self, json_folder):
        """
        Khởi tạo VerifyFrame với các embedding đã biết và tên tương ứng từ các file JSON.
        :param json_folder: Thư mục chứa các file JSON với embeddings và tên người.
        """
        self.known_data = {}
        self.name = []

        # Load embeddings từ các file JSON
        self.load_known_embeddings(json_folder)

    def load_known_embeddings(self, json_folder):
        """
        Load các embedding và tên từ thư mục chứa file JSON.
        :param json_folder: Thư mục chứa các file JSON với embeddings.
        """
        for json_file in os.listdir(json_folder):
            if json_file.endswith(".json"):
                json_path = os.path.join(json_folder, json_file)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    name = data['name']
                    embeddings = data['embeddings']

                    # Lưu các embedding của từng người trong dictionary
                    if name not in self.known_data:
                        self.known_data[name] = embeddings
                    else:
                        self.known_data[name].extend(embeddings)

    def verify(self, unknown_embeddings):
        """
        So sánh các embedding chưa biết với embedding đã biết.
        :param unknown_embeddings: List các embedding chưa biết.
        :return: List các tên tương ứng nếu khớp, nếu không thì trả về "Unknown".
        """

        self.name = []

        for unknown_embedding in unknown_embeddings:
            name_found = "Unknown"

            for name, embeddings in self.known_data.items():
                # So sánh từng embedding đã biết của một người
                matches = face_recognition.compare_faces(embeddings, unknown_embedding, tolerance=0.4)

                if True in matches:
                    # Nếu có kết quả khớp, gán tên và thoát khỏi vòng lặp
                    name_found = name
                    break

            self.name.append(name_found)

        return self.name
    
    def get_name(self):
        return self.name
    
    




