from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model
import pygame
import threading
import os
from twilio.rest import Client
import time
from flask import jsonify


app = Flask(__name__)

# Initialize pygame mixer
pygame.mixer.init()

# Load trained model
model = load_model(r"D:\pyn\emodrive\emotion_model.keras")

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Twilio Config (add your credentials obtained from twilio)
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_PHONE_NUMBER = ""
EMERGENCY_CONTACT = ""

# Audio paths
alert_sound = r"D:\pyn\emodrive\static\siren-alert-96052.mp3"
music_file = r"D:\pyn\emodrive\static\pure-love-304010.mp3"
alert_playing = False
music_playing = False
stop_detection = False

# Shared state
last_emotion = "Neutral"
last_confidence = 0
sms_sent_status = "No"
sound_triggered_status = "No"
emotion_log = []

# Send SMS
def send_sms(message_text, emotion_type=None):
    global sms_sent_status
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=message_text,
        from_=TWILIO_PHONE_NUMBER,
        to=EMERGENCY_CONTACT
    )
    timestamp = time.strftime('%H:%M:%S')
    if emotion_type:
        sms_sent_status = f"{emotion_type} alert sent at {timestamp}"
    else:
        sms_sent_status = f"SMS sent at {timestamp}"

# Make a call via Twilio (real phone call)
def make_call():
    print(f"📞 Initiating call to {EMERGENCY_CONTACT} using Windows dialer.")
    os.system(f"start tel:{EMERGENCY_CONTACT}")
    


# Play sound
def play_sound(file, sound_type):
    global alert_playing, music_playing, sound_triggered_status

    if sound_type == "alert" and alert_playing:
        return
    if sound_type == "music" and music_playing:
        return

    stop_sound()

    if sound_type == "alert":
        alert_playing = True
        sound_triggered_status = "Alert"
    elif sound_type == "music":
        music_playing = True
        sound_triggered_status = "Music"

    def sound_thread():
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()

    threading.Thread(target=sound_thread, daemon=True).start()

# Stop sound
def stop_sound():
    global alert_playing, music_playing, sound_triggered_status
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    alert_playing = False
    music_playing = False
    sound_triggered_status = "No"

# Video stream generator
def generate_frames():
    global stop_detection, last_emotion, last_confidence, emotion_log, sms_sent_status
    cap = cv2.VideoCapture(0)

    while True:
        if stop_detection:
            break

        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
            face = face / 255.0

            preds = model.predict(face)
            emotion_index = np.argmax(preds)
            emotion = emotion_labels[emotion_index]
            confidence = int(np.max(preds) * 100)

            last_emotion = emotion
            last_confidence = confidence
            emotion_log.append({
                'time': time.strftime('%H:%M:%S'),
                'emotion': emotion,
                'confidence': confidence
            })

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({confidence}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Handle emotion-specific actions
            if emotion == "Angry":
                play_sound(alert_sound, "alert")
                #send_sms("🚨 Driver appears angry!", emotion_type="Angry")
            elif emotion == "Sad":
                play_sound(music_file, "music")
                sms_sent_status = "No"
            elif emotion == "Fear":
                stop_sound()
                make_call()
                send_sms("⚠️ Driver appears fearful! Immediate attention needed!", emotion_type="Fear")
                threading.Thread(target=make_call, daemon=True).start()
            else:
                stop_sound()
                sms_sent_status = "No"

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template(
        'index.html',
        current_emotion=last_emotion,
        sms_sent=sms_sent_status,
        sound_triggered=sound_triggered_status,
        emotion_confidence=last_confidence,
        emotion_log=emotion_log[-20:]  # Show last 20 entries
    )

@app.route('/status')
def status():
    return jsonify({
        'current_emotion': last_emotion,
        'sms_sent': sms_sent_status,
        'sound_triggered': sound_triggered_status,
        'emotion_confidence': last_confidence,
        'emotion_log': emotion_log[-20:]
    })


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global stop_detection
    stop_detection = False
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    global stop_detection
    stop_detection = True
    stop_sound()
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("Starting EmoDrive Flask app...")
    app.run(debug=True)
