import cv2
import numpy as np
import tensorflow as tf


loaded_model = tf.keras.models.load_model("my_cnn_model.h5")

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (48, 48))
    input_frame = np.expand_dims(frame_resized, axis=0)
    input_frame = input_frame / 255.0  # Normalize pixel values
    return input_frame

cap = cv2.VideoCapture(0)

cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)

while True:

    ret, frame = cap.read()

    input_frame = preprocess_frame(frame)
    predictions = emotion_model.predict(input_frame)

    predicted_emotion_index = np.argmax(predictions[0])
    predicted_emotion = emotions[predicted_emotion_index]

    cv2.putText(frame, f'Emotion: {predicted_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
