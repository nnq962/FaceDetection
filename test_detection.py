import extract_embeddings
import cv2
import face_recognition
import time
import face_detection

ssd = face_detection.SSDFaceDetectorOpenCV()

image_path = "photo_test/10person.jpg"
image = cv2.imread(image_path)

image_f = face_recognition.load_image_file(image_path)
location_f = face_recognition.face_locations(image_f)

location = ssd.detect_faces(image)

print("SSD: ", len(location), location)
print("Face Recognition: ", len(location_f), location_f)

def draw_faces(frame, boxes):
    """
    Vẽ các bounding box khuôn mặt lên frame và thêm tên người.
    :param frame: Frame từ camera.
    :param boxes: Danh sách tọa độ các bounding box khuôn mặt.
    :return: Frame với bounding box khuôn mặt và tên người.
    """
    converted_locations = [(top, right, bottom, left) for left, top, right, bottom in boxes]
    # print(converted_locations)

    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        print(x1, y1, x2, y2)

        # Vẽ bounding box quanh khuôn mặt
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Gắn tên tạm thời "Test Name" lên bounding box
        name = "Test Name"
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (x1, y1 - 10), font, font_scale, (0, 0, 255), 1)
    
    cv2.imshow('Image with Face Detection', frame)
    cv2.waitKey(0)


# draw_faces(image, location_f)
draw_faces(image, location)