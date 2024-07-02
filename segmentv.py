import cv2
import torch
import numpy as np

def Segment_video(video_path, output_path):
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the YOLO model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True).to(device)
    model.eval()
    
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Obtain video properties for the output
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object for the output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB (YOLO model expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = model(frame_rgb, size=640)
        
        # Extract detections
        detections = results.xyxy[0].cpu()  # Move detections to CPU (if needed)
        people_detections = detections[detections[:, -1] == 0]  # Filter for people
        
        # Create a mask to erase non-people objects
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        
        for detection in people_detections:
            box = detection[:4].to(torch.int).numpy()
            cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)
        
        # Apply the mask to the frame
        frame[mask == 0] = 0
        
        # Write the processed frame to the output video
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    print("Video editing completed")
    
    
    

    