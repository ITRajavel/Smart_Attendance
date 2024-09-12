import cv2
import os

# Set up parameters
person_name = "Surya"  # Replace with the name of the person
dataset_path = "dataset"  # The folder where images will be saved
num_images = 100  # Number of images to capture

# Create the directory for the person if it doesn't exist
person_dir = os.path.join(dataset_path, person_name)
os.makedirs(person_dir, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

# Set the starting image count
count = 0

print("Press 'q' to quit capturing images early.")
print("Capturing images...")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image. Exiting...")
        break

    # Display the frame in a window
    cv2.imshow("Capture", frame)

    # Save every nth frame (you can adjust this to capture images less frequently)
    if count % 5 == 0:  # Adjust if needed
        image_path = os.path.join(person_dir, f"{person_name}_{count // 5:03d}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")

    # Increment the frame count
    count += 1

    # Stop if we reach the desired number of images
    if count // 5 >= num_images:
        print("Finished capturing images.")
        break

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
