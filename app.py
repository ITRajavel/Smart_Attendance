from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pygame  # Import pygame for sound playback

app = Flask(__name__)

# Initialize pygame mixer for playing sound
pygame.mixer.init()

# Function to play sound
def play_sound():
    pygame.mixer.music.load('ding.mp3')  # Ensure you have 'ding.mp3' in the same directory or provide an absolute path
    pygame.mixer.music.play()

# Step 1: Preprocess Image for Better Accuracy
def preprocess_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_rgb

# Step 2: Load and Encode Multiple Images per Person
def encode_faces(directory='faces'):
    encoded_faces = []
    names = []

    for person_name in os.listdir(directory):
        person_path = os.path.join(directory, person_name)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                img_preprocessed = preprocess_image(img)
                encodes = face_recognition.face_encodings(img_preprocessed)
                if encodes:
                    encoded_faces.append(encodes[0])
                    names.append(person_name)
    return encoded_faces, names

# Step 3: Mark Attendance
def mark_attendance(name):
    # Create a new filename based on the current date
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f'attendance_{date_str}.csv'

    # Initialize attendance data
    attendance_data = {'name': name, 'datetime': '', 'status': 'Already Marked'}

    # Open the attendance file
    with open(filename, 'a+') as f:
        # Read the existing data
        f.seek(0)
        all_data = f.readlines()
        recorded_names = [line.split(',')[0] for line in all_data]

        # Mark attendance if not already marked
        if name not in recorded_names:
            now = datetime.now()
            dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'{name},{dt_string}\n')
            attendance_data['datetime'] = dt_string
            attendance_data['status'] = 'Marked'
            play_sound()  # Play sound when attendance is marked

    return attendance_data

encoded_faces, names = encode_faces()

# Generate video stream
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        faces_in_frame = face_recognition.face_locations(small_frame)
        encodes_in_frame = face_recognition.face_encodings(small_frame, faces_in_frame)

        for encode_face, face_loc in zip(encodes_in_frame, faces_in_frame):
            matches = face_recognition.compare_faces(encoded_faces, encode_face, tolerance=0.4)
            face_distances = face_recognition.face_distance(encoded_faces, encode_face)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = names[best_match_index].upper()
                attendance_data = mark_attendance(name)
                
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} - {attendance_data['status']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance_data')
def attendance_data():
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f'attendance_{date_str}.csv'

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            attendance_list = f.readlines()
    else:
        attendance_list = []

    return jsonify(attendance_list)

if __name__ == "__main__":
    app.run(debug=True)
