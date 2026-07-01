import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mask_detector.keras")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not found!")
    exit()

print("Press 'q' to Exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (128, 128))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)[0][0]

        print(f"Prediction: {prediction:.4f}")

        if prediction >= 0.5:
            label = "With Mask"
            color = (0, 255, 0)
        else:
            label = "Without Mask"
            color = (0, 0,255 )

        confidence = prediction if prediction >= 0.5 else (1 - prediction)

        text = f"{label} : {confidence*100:.1f}%"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Face Mask Detection", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()