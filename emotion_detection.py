import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = DeepFace.analyze(frame, actions=["emotion"])

    
    emotions = result[0]['emotion']
    dominant_emotion = max(emotions, key=emotions.get)

    cv2.putText(frame, dominant_emotion, (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Emotion Analysis", frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


