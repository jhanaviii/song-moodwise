import cv2
import numpy as np
from tensorflow.keras.models import load_model

class VideoCamera:
    def __init__(self):
        self.video = None
        self.face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        self.model = load_model('model/mood_model.h5')
        self.categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Calm']
        self.stopped = False

    def start_camera(self):
        self.video = cv2.VideoCapture(0)
        self.stopped = False

    def stop_camera(self):
        self.stopped = True
        if self.video is not None:
            self.video.release()
            self.video = None

    def __del__(self):
        self.stop_camera()

    def get_frame(self):
        if self.stopped or self.video is None:
            return None, None
        success, image = self.video.read()
        if not success:
            return None, None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        mood = 'unknown'
        for (x, y, w, h) in faces:
            face = image[y:y + h, x:x + w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = face.reshape(1, 48, 48, 1)
            face = face / 255.0
            prediction = self.model.predict(face)
            mood = self.categories[np.argmax(prediction)]
            cv2.putText(image, mood, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), mood
