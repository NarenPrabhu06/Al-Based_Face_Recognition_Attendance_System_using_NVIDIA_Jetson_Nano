from flask import Flask, Response
import cv2

from recognizer import recognize
from attendance import mark_attendance

app = Flask(__name__)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise Exception("Cannot open camera")


def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        results = recognize(frame)

        for person in results:

            x, y, w, h = person["box"]
            name = person["name"]
            confidence = person["confidence"]

            if name != "Unknown":
                mark_attendance(name)
                color = (0,255,0)
            else:
                color = (0,0,255)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

            text = "{} {:.1f}".format(name, confidence)

            cv2.putText(frame,
                        text,
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame +
               b'\r\n')


@app.route("/")
def index():

    return """
    <html>

    <head>

    <title>AI Attendance System</title>

    </head>

    <body style="background:#202124;color:white;text-align:center;">

    <h1>AI Attendance System - Jetson Nano</h1>

    <img src="/video" width="800">

    </body>

    </html>
    """


@app.route("/video")
def video():

    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":

    app.run(host="0.0.0.0",
            port=5000,
            threaded=True)
