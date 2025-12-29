import cv2
import torch
import numpy as np
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from filterpy.kalman import KalmanFilter

# Configuration
WEIGHTS_PATH = "train15/weights/best.pt"
CONF_THRESHOLD = 0.40  # Confidence threshold
IOU_THRESHOLD = 0.40   # NMS IOU threshold
IMG_SIZE = 640        # Input image size for inference

# Display settings
DISPLAY_WIDTH = 1280   # Custom display width
DISPLAY_HEIGHT = 720   # Custom display height

# Custom class names for weapon detection
CLASSES = ['Weapon']

# BoT-SORT Tracking parameters
MAX_AGE = 60          # Frames to keep alive when not detected
MIN_HITS = 3          # Minimum hits before track is confirmed
IOU_THRESHOLD_TRACK = 0.2  # IOU threshold for matching

def linear_assignment(cost_matrix):
    """Linear assignment using Hungarian algorithm."""
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    """Compute IOU between two bboxes in the form [x1,y1,x2,y2]."""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    """Convert [x1,y1,x2,y2] to [x,y,s,r] where x,y is center, s is scale/area, r is aspect ratio."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """Convert [x,y,s,r] to [x1,y1,x2,y2]."""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))

class KalmanBoxTracker:
    """Kalman Filter based tracker for bounding boxes."""
    count = 0
    
    def __init__(self, bbox, cls, conf):
        """Initialize tracker with detection."""
        # State: [x, y, s, r, vx, vy, vs]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cls = cls
        self.conf = conf
        self.conf_history = deque(maxlen=30)
        self.conf_history.append(conf)
        
    def update(self, bbox, cls, conf):
        """Update tracker with new detection."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.cls = cls
        self.conf = conf
        self.conf_history.append(conf)
        
    def predict(self):
        """Advance state and return predicted bbox."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]
        
    def get_state(self):
        """Return current bbox estimate."""
        return convert_x_to_bbox(self.kf.x)
    
    def get_confidence(self):
        """Return average confidence from recent detections."""
        return np.mean(self.conf_history) if self.conf_history else self.conf

class BoTSORT:
    """BoT-SORT: Robust Associations Multi-Pedestrian Tracking."""
    
    def __init__(self, max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD_TRACK):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, detections):
        """Update tracks with new detections.
        
        Args:
            detections: numpy array of detections [x1, y1, x2, y2, conf, cls]
        
        Returns:
            tracked_objects: dict of {id: (bbox, conf, cls)}
        """
        self.frame_count += 1
        
        # Handle empty detections
        if len(detections) == 0:
            detections = np.empty((0, 6))
        elif detections.ndim == 1:
            detections = detections.reshape(1, -1)
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = {}
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :4], 
                                       int(detections[m[0], 5]), 
                                       detections[m[0], 4])
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i, :4], 
                                   int(detections[i, 5]), 
                                   detections[i, 4])
            self.trackers.append(trk)
        
        # Return tracked objects
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            
            # Only return tracks that have been hit enough times and are still alive
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or 
                                                 self.frame_count <= self.min_hits):
                ret[trk.id] = (
                    (int(d[0]), int(d[1]), int(d[2]), int(d[3])),
                    trk.get_confidence(),
                    trk.cls
                )
            i -= 1
            
            # Remove dead tracklets
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        return ret
    
    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """Assign detections to tracked objects using IOU."""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        # Handle case where detections or trackers might be empty or 1D
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.array([]), np.arange(len(trackers))
        
        # Ensure 2D arrays
        if detections.ndim == 1:
            detections = detections.reshape(1, -1)
        if trackers.ndim == 1:
            trackers = trackers.reshape(1, -1)
            
        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class DetectionLogger:
    """Log detections to file with timestamps."""
    
    def __init__(self, log_dir="detection_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_start = datetime.now()
        self.log_file = self.log_dir / f"detection_log_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        self.detections = []
        
    def log_detection(self, frame_number, timestamp, tracked_objects, source):
        """Log detection event."""
        if len(tracked_objects) == 0:
            return
            
        entry = {
            "frame": frame_number,
            "timestamp": timestamp.isoformat(),
            "source": source,
            "tracked_objects": []
        }
        
        for track_id, (bbox, conf, cls) in tracked_objects.items():
            x1, y1, x2, y2 = bbox
            entry["tracked_objects"].append({
                "track_id": track_id,
                "class": CLASSES[cls] if cls < len(CLASSES) else f"class{cls}",
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2]
            })
        
        self.detections.append(entry)
        
    def save(self):
        """Save log to file."""
        with open(self.log_file, 'w') as f:
            json.dump({
                "session_start": self.session_start.isoformat(),
                "session_end": datetime.now().isoformat(),
                "total_frames": len(self.detections),
                "tracking_algorithm": "BoT-SORT",
                "detections": self.detections
            }, f, indent=2)
        print(f"📝 Log saved to: {self.log_file}")

def load_model(weights_path):
    """Load YOLOv9 model with GPU support."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        model.to(device)
        print(f"✅ Ultralytics YOLO model loaded: {weights_path}")
        return model, device, 'ultralytics'
    except ImportError:
        model = torch.hub.load('WongKinYiu/yolov9', 'custom', path=weights_path, force_reload=False)
        model.to(device)
        print(f"✅ Torch.hub YOLO model loaded: {weights_path}")
        return model, device, 'torchhub'

def extract_detections(results, model_type):
    """Extract detections in consistent format [x1,y1,x2,y2,conf,cls]."""
    if model_type == 'ultralytics':
        if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            return torch.cat([
                boxes.xyxy,
                boxes.conf.unsqueeze(1),
                boxes.cls.unsqueeze(1)
            ], dim=1).cpu().numpy()
    else:
        if len(results.xyxy[0]) > 0:
            return results.xyxy[0].cpu().numpy()
    return np.array([])

def resize_maintain_aspect(frame, target_width=DISPLAY_WIDTH, target_height=DISPLAY_HEIGHT):
    """Resize frame while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    display_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    display_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return display_frame, (x_offset, y_offset, scale)

def draw_tracked_boxes(frame, tracked_objects):
    """Draw bounding boxes with tracking IDs - RED box, WHITE text on BLACK background."""
    if len(tracked_objects) == 0:
        return frame
    
    for track_id, (bbox, conf, cls) in tracked_objects.items():
        x1, y1, x2, y2 = bbox
        
        # Get class name
        label = CLASSES[cls] if cls < len(CLASSES) else f'class{cls}'
        
        # RED bounding box (thicker for better visibility)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Label with ID, class, and confidence
        label_text = f'ID:{track_id} {label} {conf:.2f}'
        (w, h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # BLACK background for text
        cv2.rectangle(frame, (x1, y1 - h - baseline - 5), (x1 + w + 10, y1), (0, 0, 0), -1)
        
        # WHITE text
        cv2.putText(frame, label_text, (x1 + 5, y1 - baseline - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def get_video_source(args):
    """Get video source based on arguments."""
    if args.webcam:
        return cv2.VideoCapture(0), "webcam"
    elif args.video:
        return cv2.VideoCapture(args.video), f"video:{args.video}"
    elif args.rtsp:
        return cv2.VideoCapture(args.rtsp), f"rtsp:{args.rtsp}"
    elif args.url:
        return cv2.VideoCapture(args.url), f"url:{args.url}"
    elif args.image:
        return args.image, f"image:{args.image}"
    else:
        return cv2.VideoCapture(0), "webcam"

def run_inference(args):
    """Run inference on specified source with BoT-SORT tracking."""
    # Load model
    model, device, model_type = load_model(WEIGHTS_PATH)
    
    # Initialize BoT-SORT tracker and logger
    tracker = BoTSORT(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD_TRACK)
    logger = DetectionLogger()
    
    print(f"🎯 Using BoT-SORT tracking algorithm")
    
    # Get video source
    cap, source_name = get_video_source(args)
    
    # Handle image source separately
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"❌ Error: Could not load image {args.image}")
            return
        
        # Run inference
        results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, 
                       imgsz=IMG_SIZE, device=device, verbose=False)
        detections = extract_detections(results, model_type)
        
        # Update tracker
        tracked_objects = tracker.update(detections)
        
        # Log detections
        logger.log_detection(0, datetime.now(), tracked_objects, source_name)
        
        # Draw boxes
        annotated_frame = draw_tracked_boxes(frame.copy(), tracked_objects)
        display_frame, _ = resize_maintain_aspect(annotated_frame, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        
        # Show image
        cv2.imshow('Detection Result - Press any key to exit', display_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        logger.save()
        return
    
    # Video source handling
    if not cap.isOpened():
        print(f"❌ Error: Could not open source: {source_name}")
        return
    
    # Set properties for webcam
    if args.webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create window
    window_name = f'YOLOv9 + BoT-SORT - {source_name} - Press Q to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    print(f"🎥 Starting inference on {source_name}")
    print("📋 Controls: 'q' = Quit, 'r' = Reset window, 's' = Save frame")
    
    # FPS calculation
    prev_time = time.time()
    frame_count = 0
    total_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of stream or failed to capture frame")
            break
        
        frame_count += 1
        total_frames += 1
        
        # Run inference
        results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, 
                       imgsz=IMG_SIZE, device=device, verbose=False)
        
        # Extract detections
        detections = extract_detections(results, model_type)
        
        # Update BoT-SORT tracker
        tracked_objects = tracker.update(detections)
        
        # Log tracked objects
        logger.log_detection(total_frames, datetime.now(), tracked_objects, source_name)
        
        # Draw tracked boxes
        annotated_frame = draw_tracked_boxes(frame, tracked_objects)
        
        # Resize with aspect ratio preservation
        display_frame, _ = resize_maintain_aspect(annotated_frame, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        
        # Calculate FPS
        curr_time = time.time()
        if curr_time - prev_time > 0:
            fps = frame_count / (curr_time - prev_time)
            frame_count = 0
            prev_time = curr_time
        else:
            fps = 0
        
        # Add info overlay with WHITE text on BLACK background
        info_lines = [
            f'FPS: {fps:.1f} | Tracker: BoT-SORT | Source: {source_name}',
            f'Active Tracks: {len(tracked_objects)} | Logged Events: {len(logger.detections)}',
            f'Frame: {total_frames} | Confidence: {CONF_THRESHOLD}'
        ]
        
        y_pos = 30
        for line in info_lines:
            (w, h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (5, y_pos - h - baseline - 5), 
                         (w + 15, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(display_frame, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += h + baseline + 10
        
        # Show frame
        cv2.imshow(window_name, display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
            print("🔄 Window size reset")
        elif key == ord('s'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"detection_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Frame saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logger.save()
    print("✅ Inference stopped cleanly.")

def main():
    parser = argparse.ArgumentParser(description='YOLOv9 Weapon Detection with BoT-SORT Tracking')
    
    # Source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--webcam', action='store_true', help='Use webcam (default)')
    source_group.add_argument('--video', type=str, help='Path to video file')
    source_group.add_argument('--image', type=str, help='Path to image file')
    source_group.add_argument('--rtsp', type=str, help='RTSP stream URL')
    source_group.add_argument('--url', type=str, help='Video URL')
    
    args = parser.parse_args()
    
    # Default to webcam if no source specified
    if not any([args.webcam, args.video, args.image, args.rtsp, args.url]):
        args.webcam = True
    
    run_inference(args)

if __name__ == "__main__":
    main()