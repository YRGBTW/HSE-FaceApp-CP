import subprocess

subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
from flask import Flask, render_template, Response, send_file, request
import cv2 as cv2
import numpy as np
from PIL import Image
import os
import logging
import time

app = Flask(__name__)

logger = logging.getLogger('RecognitionLogger')
logger.setLevel(logging.INFO)

log_file = 'log.txt'
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

last_recognition_times = {}


def face_recognition():
    current_time = time.time()

    cascadePath = "haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascadePath)
    recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 123)
    recognizer.setRadius(1)
    recognizer.setNeighbors(8)
    recognizer.setGridX(8)
    recognizer.setGridY(8)
    recognizer.setThreshold(123)

    def get_images(path):

        image_paths = [os.path.join(path, f) for f in os.listdir(path)]

        images = []
        labels = []
        names = []
        subject_number = 0

        for image_path in image_paths:
            gray = Image.open(image_path).convert('L')
            image = np.array(gray, 'uint8')
            subject_number += 1
            subject_name = os.path.split(image_path)[1].replace(".jpg", "")
            faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(subject_number)
                names.append(subject_name)
        return images, labels, names

    path = 'photos'

    images, labels, names = get_images(path)

    if (len(images) != 0 and len(labels) != 0):
        recognizer.train(images, np.array(labels))

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                image = np.array(gray, 'uint8')
                faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cord = x, y - 10
                    font = cv2.FONT_HERSHEY_COMPLEX
                    fontScale = 0.5
                    color = (0, 255, 0)
                    thickness = 1
                    if conf < 51 and number_predicted != -1:
                        name = names[number_predicted - 1]
                        text = name
                        time_difference = 1000
                        if name in last_recognition_times:
                            last_recognition_time = last_recognition_times[name]
                            time_difference = current_time - last_recognition_time

                        if time_difference > 300:  # 5 минут = 300 секунд
                            logger.info(f'Лицо распознано: {name}')
                            last_recognition_times[name] = current_time
                    else:
                        text = 'Не распознано'
                        color = (0, 0, 255)

                    cv2.putText(frame, text, cord, font, fontScale, color, thickness)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/download_info')
def download_chm():
    pdf_file_path = 'info.pdf'
    return send_file(pdf_file_path, as_attachment=True)


@app.route('/download_log')
def download_log():
    return send_file(log_file, as_attachment=True)


@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' in request.files:
        photo = request.files['photo']
        photo.save(os.path.join('photos', photo.filename))

        return 'Фото успешно загружено!'
    return 'Фото не найдено в запросе.'


if __name__ == "__main__":
    app.run(debug=False)
