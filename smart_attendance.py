import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Step 1: Preprocess Image for Better Accuracy
def preprocess_image(img):
    # Convert image to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Apply Histogram Equalization to improve contrast
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_rgb

# Step 2: Load and Encode Multiple Images per Person
def encode_faces(directory='faces'):
    encoded_faces = []
    names = []
    
    # Iterate through each person in the dataset directory
    for person_name in os.listdir(directory):
        person_path = os.path.join(directory, person_name)
        
        # Ensure the path is a directory (i.e., a person folder)
        if os.path.isdir(person_path):
            # Iterate through each image in the person's directory
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                
                # Preprocess image to improve accuracy
                img_preprocessed = preprocess_image(img)
                
                # Get face encodings for each image
                encodes = face_recognition.face_encodings(img_preprocessed)
                if encodes:
                    encoded_faces.append(encodes[0])
                    names.append(person_name)
    
    return encoded_faces, names

# Step 3: Mark Attendance
def mark_attendance(name):
    with open('attendance.csv', 'r+') as f:
        all_data = f.readlines()
        recorded_names = [line.split(',')[0] for line in all_data]
        if name not in recorded_names:
            now = datetime.now()
            dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'\n{name},{dt_string}')

# Main Function to Recognize Faces
def main():
    # Load and Encode known faces
    encoded_faces, names = encode_faces()

    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces and encode them
        faces_in_frame = face_recognition.face_locations(small_frame)
        encodes_in_frame = face_recognition.face_encodings(small_frame, faces_in_frame)

        # Compare detected faces with known faces
        for encode_face, face_loc in zip(encodes_in_frame, faces_in_frame):
            matches = face_recognition.compare_faces(encoded_faces, encode_face, tolerance=0.4)
            face_distances = face_recognition.face_distance(encoded_faces, encode_face)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = names[best_match_index].upper()
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                mark_attendance(name)

        cv2.imshow('Smart Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
