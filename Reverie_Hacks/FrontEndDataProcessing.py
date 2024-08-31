import cv2
import os
from ultralytics import YOLO
from datetime import datetime

# Load the YOLOv8 model
model = YOLO('yolov8m-seg.pt')

# Directories
video_directory = "FrontEndVideo"  # Directory where the input image is stored
output_dir = 'yolo_preds'  # Directory to save annotated frames

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def segment_and_annotate_images(frame):
    """Process a frame with the YOLO model and save the annotated image."""
    preds = model.predict([frame])  # Predict on the input frame

    for pred in preds:
        frame_with_overlay = pred.plot()  # Plot predictions (draw bounding boxes on the frame)

        # Save the annotated image
        annotated_img_name = f"annotated_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        annotated_img_path = os.path.join(output_dir, annotated_img_name)
        cv2.imwrite(annotated_img_path, frame_with_overlay)
        print(f"Annotated image saved at: {annotated_img_path}")

# Load the image from FrontEndVideo directory
image_filename = "Signs of a Stroke - New.jpeg"  # Replace with the actual image file name
image_path = os.path.join(video_directory, image_filename)
frame = cv2.imread(image_path)

if frame is not None:
    segment_and_annotate_images(frame)
else:
    print(f"Error: Unable to load image from {image_path}")
