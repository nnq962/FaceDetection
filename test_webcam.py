import cv2
import requests
import numpy as np

# URL của HTTP stream
url = 'http://192.168.1.136:8889/cam'

# Gửi yêu cầu HTTP
stream = requests.get(url, stream=True)

print("hahahaha")

# Đọc và xử lý từng khung hình
if stream.status_code == 200:
    byte_stream = bytes()
    for chunk in stream.iter_content(chunk_size=1024):
        byte_stream += chunk
        print(byte_stream[:5])
        # Tìm vị trí bắt đầu và kết thúc của một frame JPEG
        start = byte_stream.find(b'\xff\xd8')
        end = byte_stream.find(b'\xff\xd9')
        print(start, end)
        if start != -1 and end != -1:

            jpg = byte_stream[start:end+2]
            byte_stream = byte_stream[end+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            print(frame)
            if frame is not None:
                cv2.imshow("Video", frame)

            if cv2.waitKey(1) == ord('q'):
                break
else:
    print("Không thể kết nối đến stream HTTP")

print("hihihi")

cv2.destroyAllWindows()