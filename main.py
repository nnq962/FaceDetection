import camera
import extract_embeddings
import cv2
import verify_frame

embedding_folder = 'embeddings_data'
cam_manager = camera.CameraManager(camera_id=0, detector="mediapipe")
verifier = verify_frame.VerifyFrame(data_folder=embedding_folder, threshold=0.25)
get_embs = extract_embeddings.FaceEmbeddingExtractor()

while True:
    frame = cam_manager.get_frame(flip_code=1)

    boxes = cam_manager.detect_faces(frame)

    embeddings = get_embs.extract_embeddings(frame, boxes)

    result = verifier.verify(embeddings)

    names = result if result else ["Unknown"] * len(boxes)

    frame_with_faces = cam_manager.draw_faces(frame, boxes, names)

    cv2.imshow('Camera with Face Detection', frame_with_faces)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_manager.release()

cv2.destroyAllWindows()
