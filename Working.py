# from ultralytics import YOLO
# import cv2

# # Load the YOLO model with the specified weights
# model = YOLO("C:\\Users\\VINJAYCAS\\Desktop\\codeprojects\\runs\\detect\\train10\\weights\\best.pt")

# # Use the model to predict from the webcam (source="0")
# results = model.predict(source="0", show=True, conf=0.5)

# # Optionally, process the results if needed
# # For example, you can access the predictions as follows:
# for result in results:
#     # Access bounding boxes, class ids, and confidences
#     boxes = result.boxes  # Get bounding boxes
#     print(boxes)  # Print the detected boxes



import cv2
import torch
from ultralytics import YOLO 
import os
import time
from ultralytics.utils.plotting import Annotator, colors


# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLOv8 model and move it to the GPU if available
model = YOLO("C:\\Users\\VINJAYCAS\\Desktop\\codeprojects\\runs\\detect\\train10\\weights\\best.pt").to(device)

# Blur ratio
blur_ratio = 50

# Detection flag and file handle
detected = False

def preprocess_frame(frame):
    
    # Resize frame to 640x640 (expected input size for YOLOv8)
    resized_frame = cv2.resize(frame, (640, 640))

    # Convert frame from (H, W, C) to (C, H, W) for YOLOv8, and add batch dimension (B, C, H, W)
    frame_tensor = torch.tensor(resized_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Move to GPU if available
    frame_tensor = frame_tensor.to(device)

    return frame_tensor

def blur_humans(frame):
    global detected

    # Perform inference on the frame using the YOLOv8 model
    results = model(frame)

    # Move results back to CPU for processing
    boxes = results[0].boxes.xyxy.cpu().tolist()  # List of bounding box coordinates
    clss = results[0].boxes.cls.cpu().tolist()    # List of class IDs (labels)
    confs = results[0].boxes.conf.cpu().tolist()  # List of confidence scores
    
    # Move results back to CPU for processing
    person_detected = any(int(cls) == 0 for cls in clss)
    
    if person_detected and not detected:
        os.startfile("C:\\Users\\VINJAYCAS\\Desktop\\codeprojects\\Detect_Vincent.png")
        detected = True
    elif not person_detected and detected:
        os.system("TASKKILL /F /IM Microsoft.Photos.exe")
        time.sleep(1) 
        detected = False

    # Move results back to CPU for processing
    for box, cls, conf in zip(boxes, clss, confs):
        if int(cls) == 0:  # 0 is the class ID for 'person' in the COCO dataset
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)

            # Extract the region of interest (the person)
            person_region = frame[y1:y2, x1:x2]

            # Blur the person's region
            blurred_person = cv2.blur(person_region, (blur_ratio, blur_ratio))

            # Place the blurred person back into the frame
            frame[y1:y2, x1:x2] = blurred_person

            # Draw the bounding box and label with confidence
            label = f"Vincent: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    return frame

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Blur detected humans in the frame
#     frame = blur_humans(frame)

#     # Display the result
#     cv2.imshow("Live Human Blurring", frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

while True:
    ret, frame = cap.read()
    if not ret:
        breakokay

    # Create a copy of the original frame (non-blurred)
    original_frame = frame.copy()

    # Blur detected humans in the original frame
    blurred_frame = blur_humans(frame)

    # Concatenate the original and blurred frames horizontally
    combined_frame = cv2.hconcat([original_frame, blurred_frame])

    # Display the result with original and blurred side by side
    cv2.imshow("Original and Blurred Humans", combined_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()