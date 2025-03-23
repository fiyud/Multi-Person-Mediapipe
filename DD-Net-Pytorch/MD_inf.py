import cv2
import torch
import numpy as np
import mediapipe as mp
import time
import os
from ultralytics import YOLO
from collections import deque

# Import DDNet model
from models.DDNet_Original import DDNet_Original
# If you're using the modified version with stats stream:
# from models.DDNet_with_stats import DDNet_with_Stats_Stream


class ActionRecognizer:
    def __init__(self, model_path, config, action_classes):
        """
        Initialize the action recognition model.
        
        Args:
            model_path: Path to the saved DDNet model weights
            config: Configuration object with model parameters
            action_classes: List of action class names
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model configuration
        self.config = config
        self.frame_l = config.frame_l  # Number of frames to collect before inference
        self.joint_n = config.joint_n  # Number of joints
        self.joint_d = config.joint_d  # Joint dimensions (2D or 3D)
        self.feat_d = config.feat_d    # JCD feature dimension
        self.filters = config.filters  # Number of filters
        self.clc_num = len(action_classes)  # Number of action classes
        
        # Initialize DDNet model
        self.model = DDNet_Original(
            self.frame_l, self.joint_n, self.joint_d, 
            self.feat_d, self.filters, self.clc_num
        )
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Store action class names
        self.action_classes = action_classes
        
        # Buffer to store skeleton data for each tracked person
        self.person_buffers = {}
        
        # Mapping from MediaPipe keypoints to DDNet expected format
        # This will need customization based on your DDNet's expected input format
        self.mediapipe_to_ddnet_mapping = self._get_mediapipe_to_ddnet_mapping()
    
    def _get_mediapipe_to_ddnet_mapping(self):
 
        mp_pose = mp.solutions.pose.PoseLandmark
        
        # Example mapping (adjust for your specific model):
        mapping = {
            0: mp_pose.NOSE,                     # face
            # MediaPipe doesn't have NECK, so we'll use the midpoint between shoulders
            1: None,                             # neck (will be calculated as midpoint between shoulders)
            # MediaPipe has a MID_HIP point that can be used for belly
            2: None,                             # belly (will be calculated as average of left and right shoulder, and hip)
            3: mp_pose.RIGHT_SHOULDER,           # right shoulder
            4: mp_pose.LEFT_SHOULDER,            # left shoulder
            5: mp_pose.RIGHT_HIP,                # right hip
            6: mp_pose.LEFT_HIP,                 # left hip
            7: mp_pose.RIGHT_ELBOW,              # right elbow
            8: mp_pose.LEFT_ELBOW,               # left elbow
            9: mp_pose.RIGHT_KNEE,               # right knee
            10: mp_pose.LEFT_KNEE,               # left knee
            11: mp_pose.RIGHT_WRIST,             # right wrist
            12: mp_pose.LEFT_WRIST,              # left wrist
            13: mp_pose.RIGHT_ANKLE,             # right ankle
            14: mp_pose.LEFT_ANKLE,              # left ankle
        }
        
        return mapping

    def _calculate_jcd_features(self, pose_sequence):
        """
        Calculate Joint-to-Joint distance (JCD) features from pose sequence.
        
        Args:
            pose_sequence: Array of shape (frame_l, joint_n, joint_d) containing pose data
            
        Returns:
            JCD features of shape (frame_l, feat_d)
        """
        M = []
        # For each frame, calculate the pairwise distances between joints
        for f in range(len(pose_sequence)):
            # Get pairwise euclidean distances between all joints
            dist_matrix = np.zeros((self.joint_n, self.joint_n))
            for i in range(self.joint_n):
                for j in range(i+1, self.joint_n):
                    dist = np.linalg.norm(pose_sequence[f, i] - pose_sequence[f, j])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
            
            # Flatten the upper triangle (excluding diagonal) to get feature vector
            upper_tri_indices = np.triu_indices(self.joint_n, k=1)
            frame_features = dist_matrix[upper_tri_indices]
            M.append(frame_features)
        
        M = np.stack(M)
        
        # Normalize features
        M = (M - np.mean(M)) / (np.std(M) + 1e-10)
        
        return M
    
    def _convert_mediapipe_to_ddnet_format(self, pose_landmarks):
        if not pose_landmarks:
            return None
        
        # Initialize skeleton array
        skeleton = np.zeros((self.joint_n, self.joint_d))
        
        # Extract landmarks
        landmarks = pose_landmarks.landmark
        
        # Extract coordinates for each joint based on the mapping
        for ddnet_idx, mp_idx in self.mediapipe_to_ddnet_mapping.items():
            if mp_idx is not None:
                # Direct mapping
                landmark = landmarks[mp_idx]
                skeleton[ddnet_idx, 0] = landmark.x
                skeleton[ddnet_idx, 1] = landmark.y
                if self.joint_d == 3:
                    skeleton[ddnet_idx, 2] = landmark.z
        
        # Calculate neck as midpoint between shoulders
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        skeleton[1, 0] = (right_shoulder.x + left_shoulder.x) / 2
        skeleton[1, 1] = (right_shoulder.y + left_shoulder.y) / 2
        if self.joint_d == 3:
            skeleton[1, 2] = (right_shoulder.z + left_shoulder.z) / 2
        
        # Calculate belly as midpoint between shoulders and hips
        right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        skeleton[2, 0] = (right_shoulder.x + left_shoulder.x + right_hip.x + left_hip.x) / 4
        skeleton[2, 1] = (right_shoulder.y + left_shoulder.y + right_hip.y + left_hip.y) / 4
        if self.joint_d == 3:
            skeleton[2, 2] = (right_shoulder.z + left_shoulder.z + right_hip.z + left_hip.z) / 4
        
        return skeleton
    
    def add_pose_to_buffer(self, track_id, pose_landmarks):
        """
        Add pose landmarks to the buffer for a specific tracked person.
        
        Args:
            track_id: ID of the tracked person
            pose_landmarks: MediaPipe pose landmarks result
        """
        # Convert landmarks to the format expected by DDNet
        skeleton = self._convert_mediapipe_to_ddnet_format(pose_landmarks)
        
        if skeleton is None:
            return
        
        # Initialize buffer for new person
        if track_id not in self.person_buffers:
            self.person_buffers[track_id] = deque(maxlen=self.frame_l)
        
        # Add skeleton to the buffer
        self.person_buffers[track_id].append(skeleton)
    
    def predict_action(self, track_id):
        """
        Predict action for a tracked person if enough frames are available.
        
        Args:
            track_id: ID of the tracked person
            
        Returns:
            action_class: Predicted action class name or None if not enough frames
            confidence: Confidence score of prediction or None
        """
        if track_id not in self.person_buffers:
            return None, None
        
        # Check if we have enough frames
        buffer = self.person_buffers[track_id]
        if len(buffer) < self.frame_l:
            return None, None
        
        # Convert buffer to numpy array
        pose_sequence = np.array(list(buffer))
        
        # Calculate JCD features
        jcd_features = self._calculate_jcd_features(pose_sequence)
        
        # Convert to PyTorch tensors
        X_0 = torch.from_numpy(jcd_features).float().unsqueeze(0)  # Add batch dimension
        X_1 = torch.from_numpy(pose_sequence).float().unsqueeze(0)  # Add batch dimension
        
        # Move tensors to the appropriate device
        X_0 = X_0.to(self.device)
        X_1 = X_1.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(X_0, X_1)
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, prediction = torch.max(prob, 1)
        
        action_class = self.action_classes[prediction.item()]
        confidence_value = confidence.item()
        
        return action_class, confidence_value


# Modify process_video function to include action recognition
def process_video_with_action(video_path, output_path, yolo_model, pose, mp_drawing, mp_pose, action_recognizer):
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
        
        results = yolo_model.track(source=frame, device=0, persist=True, tracker="bytetrack.yaml", classes=[0])[0]
        new_frame = frame.copy()
        
        if results.boxes is not None:
            for track in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls, track_id = map(int, track)
                if cls == 0:  # Person class
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue
                    
                    person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(person_rgb)
                    
                    if pose_results.pose_landmarks:
                        # Draw pose landmarks on the crop
                        mp_drawing.draw_landmarks(person_crop, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        
                        # Add pose to action recognizer buffer
                        action_recognizer.add_pose_to_buffer(track_id, pose_results.pose_landmarks)
                        
                        # Try to predict action
                        action, confidence = action_recognizer.predict_action(track_id)
                        
                    new_frame[y1:y2, x1:x2] = person_crop
                    
                    # Draw bounding box and ID
                    cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(new_frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add action prediction if available
                    if action and confidence:
                        cv2.putText(new_frame, f'Action: {action} ({confidence:.2f})', 
                                   (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        out.write(new_frame)
    
    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")


# Define a simple config class to match DDNet's expectations
class SimpleConfig:
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters):
        self.frame_l = frame_l
        self.joint_n = joint_n
        self.joint_d = joint_d
        self.feat_d = feat_d
        self.filters = filters


def main(input_path, output_folder=None, model_path=None):
    """Main function to process input with YOLO, MediaPipe and DDNet."""
    # Check if input path exists
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    # Set default model_path if not provided
    if model_path is None:
        model_path = "D:\\NCKHSV.2024-2025\\MultiMediapipe\\DD-Net-Pytorch\\weights\\JHMDB_latest_epoch_600.pt"
    
    # Define action classes based on your dataset (example for JHMDB)
    action_classes = [
        "brush_hair", "catch", "clap", "climb_stairs", "golf", 
        "jump", "kick_ball", "pick", "pour", "pullup", "push", 
        "run", "shoot_ball", "shoot_bow", "shoot_gun", "sit", 
        "stand", "swing_baseball", "throw", "walk", "wave"
    ]  # Adjust this list to match your model's classes
        
    try:
        print(f"Loading YOLO model...")
        yolo_model = YOLO('yolov8x.pt')  # Changed to standard model name
        print(f"YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Create DDNet config
    config = SimpleConfig(
        frame_l=32,  # Number of frames to collect
        joint_n=15,  # Number of joints for JHMDB
        joint_d=2,   # 2D coordinates
        feat_d=105,  # JCD feature dimension (15 choose 2 = 105)
        filters=64   # Number of filters in DDNet
    )
    
    # Initialize action recognizer
    print(f"Loading DDNet model from {model_path}...")
    action_recognizer = ActionRecognizer(model_path, config, action_classes)
    print(f"DDNet model loaded successfully")
    
    if is_directory(input_path):
        # Handle directory processing
        process_directory_with_action(input_path, yolo_model, pose, mp_drawing, mp_pose, action_recognizer, output_folder)
    elif is_video_file(input_path):
        # Handle video processing
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, 'output_' + os.path.basename(input_path))
        else:
            output_path = 'output_' + os.path.basename(input_path)
        
        process_video_with_action(input_path, output_path, yolo_model, pose, mp_drawing, mp_pose, action_recognizer)
    else:
        print(f"Error: Unsupported file type or not a video: {input_path}")
    
    cv2.destroyAllWindows()


# Helper functions from the original code
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


# Similar to process_directory but for action recognition - implement as needed
def process_directory_with_action(directory_path, yolo_model, pose, mp_drawing, mp_pose, action_recognizer, output_folder=None):
    """Process all videos in a directory with action recognition."""
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
            if is_video_file(file_path):
                output_path = os.path.join(output_folder, 'output_' + filename)
                process_video_with_action(file_path, output_path, yolo_model, pose, mp_drawing, mp_pose, action_recognizer)
                processed_count += 1
            # Note: For action recognition, we process only videos, not images
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            errors_count += 1
    
    print(f"Processed {processed_count} videos from {directory_path}")
    if errors_count > 0:
        print(f"Encountered errors in {errors_count} files")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Process videos with YOLO, MediaPipe, and DDNet action recognition.')
    parser.add_argument('--input', '-i', dest='input_path', help='Path to video or folder')
    parser.add_argument('--output', '-o', dest='output_folder', help='Path to output folder')
    parser.add_argument('--model', '-m', dest='model_path', help='Path to DDNet model weights')
    
    args = parser.parse_args()
    
    input_path = args.input_path or r"C:\Users\ADMIN\Downloads\241481547_main_xxl.mp4"
    output_folder = args.output_folder or r"D:\NCKHSV.2024-2025\MultiMediapipe\out"
    model_path = args.model_path or r"D:\NCKHSV.2024-2025\MultiMediapipe\DD-Net-Pytorch\weights\JHMDB_latest_epoch_600.pt"
    
    print(f"Starting processing with:")
    print(f"- Input: {input_path} (exists: {os.path.exists(input_path)})")
    print(f"- Output folder: {output_folder}")
    print(f"- Model path: {model_path} (exists: {os.path.exists(model_path)})")
    
    main(input_path, output_folder, model_path)