import cv2
import torch
import numpy as np
import mediapipe as mp
import time
import os
from ultralytics import YOLO

import cv2
import torch
import numpy as np
import mediapipe as mp
import time
import os
from ultralytics import YOLO

def is_image_file(file_path):
    """Check if the input file is an image based on extension."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions

def is_video_file(file_path):
    """Check if the input file is a video based on extension."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions

def is_directory(path):
    """Check if path is a directory."""
    return os.path.isdir(path)

def process_directory(directory_path, yolo_model, pose, mp_drawing, mp_pose, output_folder=None):
    """Process all images and videos in a directory."""
    if output_folder is None:
        output_folder = os.path.join(directory_path, 'output')
    
    os.makedirs(output_folder, exist_ok=True)
    
    processed_count = 0
    errors_count = 0
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Skip directories and non-file items
        if not os.path.isfile(file_path):
            continue
            
        print(f"Processing {filename}...")
        
        try:
            if is_image_file(file_path):
                process_image(file_path, yolo_model, pose, mp_drawing, mp_pose, output_folder)
                processed_count += 1
            elif is_video_file(file_path):
                output_path = os.path.join(output_folder, 'output_' + filename)
                process_video(file_path, output_path, yolo_model, pose, mp_drawing, mp_pose)
                processed_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            errors_count += 1
    
    print(f"Processed {processed_count} files from {directory_path}")
    if errors_count > 0:
        print(f"Encountered errors in {errors_count} files")

def process_image(image_path, yolo_model, pose, mp_drawing, mp_pose, output_folder=None):
    """Process a single image with YOLO detection and MediaPipe pose estimation."""
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Process the frame
    results = yolo_model.track(source=frame, persist=True, tracker="bytetrack.yaml", classes=[0])[0]
    output_frame = frame.copy()
    
    if results.boxes is not None:
        for track in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls, track_id = map(int, track)
            if cls == 0:
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                    
                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(person_rgb)
                
                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(person_crop, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                output_frame[y1:y2, x1:x2] = person_crop
                
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, 'output_' + os.path.basename(image_path))
    else:
        output_path = 'output_' + os.path.basename(image_path)
    
    cv2.imwrite(output_path, output_frame)
    print(f"Processed image saved to {output_path}")
    
    return output_frame

def process_video(video_path, output_path, yolo_model, pose, mp_drawing, mp_pose):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        results = yolo_model.track(source=frame, persist=True, tracker="bytetrack.yaml", classes=[0])[0]
        new_frame = frame.copy()
        
        if results.boxes is not None:
            for track in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls, track_id = map(int, track)
                if cls == 0:
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue
                    
                    person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(person_rgb)
                    
                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(person_crop, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    new_frame[y1:y2, x1:x2] = person_crop
                    
                    cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(new_frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(new_frame)
    
    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

def main(input_path, output_folder=None):
    """Main function to process input path (file or directory)."""
    # Check if input path exists
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        return
        
    try:
        print(f"Loading YOLO model...")
        yolo_model = YOLO('yolo12s.pt')
        print(f"YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    
    if is_directory(input_path):
        process_directory(input_path, yolo_model, pose, mp_drawing, mp_pose, output_folder)
    elif is_image_file(input_path):
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        process_image(input_path, yolo_model, pose, mp_drawing, mp_pose, output_folder)
    elif is_video_file(input_path):
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, 'output_' + os.path.basename(input_path))
        else:
            output_path = 'output_' + os.path.basename(input_path)
        
        process_video(input_path, output_path, yolo_model, pose, mp_drawing, mp_pose)
    else:
        print(f"Error: Unsupported file type: {input_path}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Process images and videos with YOLO and MediaPipe.')
    parser.add_argument('--input', '-i', dest='input_path', help='Path to image, video, or folder')
    parser.add_argument('--output', '-o', dest='output_folder', help='Path to output folder')
    
    args = parser.parse_args()
    
    # Default values if not provided
    input_path = args.input_path or r"D:\NCKHSV.2024-2025\MultiMediapipe\dance.mp4"
    output_folder = args.output_folder or r"D:\NCKHSV.2024-2025\MultiMediapipe\out"
    
    print(f"Starting processing with:")
    print(f"- Input: {input_path} (exists: {os.path.exists(input_path)})")
    print(f"- Output folder: {output_folder}")
    
    main(input_path, output_folder)