"""
Enhanced Weapon Detection System v4.0 - Streamlit Edition
Advanced false alarm reduction with temporal filtering, size validation, and confidence smoothing
Run with: streamlit run weapon_detection_streamlit.py
"""

import cv2
import numpy as np
import torch
import time
import streamlit as st
from datetime import datetime
from collections import deque
from filterpy.kalman import KalmanFilter
import threading
import queue

# Page configuration
st.set_page_config(
    page_title="🛡️ Weapon Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #1a1a2e;
    }
    .stApp {
        background-color: #1a1a2e;
    }
    h1, h2, h3, h4, h5, h6, p, label {
        color: #ecf0f1 !important;
    }
    .stMetric {
        background-color: #2c3e50;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #34495e;
    }
    .stMetric label {
        color: #95a5a6 !important;
        font-size: 14px !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #3498db !important;
        font-size: 32px !important;
    }
    .sidebar .sidebar-content {
        background-color: #16213e;
    }
    div[data-testid="stSidebar"] {
        background-color: #16213e;
    }
    .stButton button {
        background-color: #2ecc71;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #27ae60;
    }
    .stop-button button {
        background-color: #e74c3c !important;
    }
    .stop-button button:hover {
        background-color: #c0392b !important;
    }
</style>
""", unsafe_allow_html=True)


# ====================== OPTIMIZED IOU CALCULATION ======================
def iou_batch(dets, trks):
    """Vectorized IOU computation for speed"""
    if len(dets) == 0 or len(trks) == 0:
        return np.zeros((len(dets), len(trks)), dtype=np.float32)

    dets = np.atleast_2d(dets).astype(np.float64)
    trks = np.atleast_2d(trks).astype(np.float64)

    xx1 = np.maximum(dets[:, 0][:, None], trks[:, 0][None, :])
    yy1 = np.maximum(dets[:, 1][:, None], trks[:, 1][None, :])
    xx2 = np.minimum(dets[:, 2][:, None], trks[:, 2][None, :])
    yy2 = np.minimum(dets[:, 3][:, None], trks[:, 3][None, :])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area_d = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])[:, None]
    area_t = (trks[:, 2] - trks[:, 0]) * (trks[:, 3] - trks[:, 1])[None, :]
    union = area_d + area_t - inter
    return inter / (union + 1e-10)


def linear_assignment(cost_matrix):
    """Hungarian algorithm for optimal assignment"""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost_matrix)
        return np.column_stack((rows, cols))
    except ImportError:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])


# ====================== FALSE ALARM REDUCTION FILTERS ======================
class ConfirmedDetectionTracker:
    """Temporal consistency filter - requires persistent detections"""
    def __init__(self, confirmation_frames=5):
        self.confirmation_frames = confirmation_frames
        self.detection_history = {}
        
    def add_detection(self, track_id, cls):
        if track_id not in self.detection_history:
            self.detection_history[track_id] = deque(maxlen=self.confirmation_frames)
        self.detection_history[track_id].append(cls)
        
    def is_confirmed(self, track_id):
        if track_id not in self.detection_history:
            return False
        history = self.detection_history[track_id]
        return len(history) >= self.confirmation_frames and len(set(history)) == 1
    
    def cleanup(self, active_ids):
        """Remove stale tracks"""
        stale = [tid for tid in self.detection_history if tid not in active_ids]
        for tid in stale:
            del self.detection_history[tid]


def filter_by_size(dets, frame_shape, min_ratio=0.003, max_ratio=0.4):
    """Remove unrealistic detection sizes"""
    if len(dets) == 0:
        return dets
    
    frame_area = frame_shape[0] * frame_shape[1]
    filtered = []
    
    for det in dets:
        x1, y1, x2, y2 = det[:4]
        bbox_area = (x2 - x1) * (y2 - y1)
        area_ratio = bbox_area / frame_area
        
        if min_ratio <= area_ratio <= max_ratio:
            filtered.append(det)
    
    return np.array(filtered) if filtered else np.empty((0, 6))


def validate_aspect_ratio(dets, min_ratio=0.15, max_ratio=6.0):
    """Filter based on realistic weapon aspect ratios"""
    if len(dets) == 0:
        return dets
    
    filtered = []
    for det in dets:
        x1, y1, x2, y2 = det[:4]
        w = x2 - x1
        h = y2 - y1
        aspect_ratio = w / (h + 1e-6)
        
        if min_ratio <= aspect_ratio <= max_ratio or min_ratio <= 1/aspect_ratio <= max_ratio:
            filtered.append(det)
    
    return np.array(filtered) if filtered else np.empty((0, 6))


class SceneContextFilter:
    """Remove edge detections that are often false positives"""
    def __init__(self, edge_margin=0.05):
        self.edge_margin = edge_margin
        
    def filter_edge_detections(self, dets, frame_shape):
        if len(dets) == 0:
            return dets
        
        h, w = frame_shape[:2]
        margin_x = w * self.edge_margin
        margin_y = h * self.edge_margin
        
        filtered = []
        for det in dets:
            x1, y1, x2, y2 = det[:4]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            if margin_x < cx < w - margin_x and margin_y < cy < h - margin_y:
                filtered.append(det)
        
        return np.array(filtered) if filtered else np.empty((0, 6))


# ====================== KALMAN TRACKER ======================
def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, w * h, w / h]).reshape((4, 1))


def convert_x_to_bbox(x):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return [x[0] - w/2, x[1] - h/2, x[0] + w/2, x[1] + h/2]


class KalmanBoxTracker:
    """Enhanced tracker with confidence smoothing"""
    count = 0
    
    def __init__(self, bbox, cls=0, conf=0.5):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.F[0, 4] = self.kf.F[1, 5] = self.kf.F[2, 6] = 1
        self.kf.H = np.zeros((4, 7))
        self.kf.H[:4, :4] = np.eye(4)
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hit_streak = 0
        self.time_since_update = 0
        self.cls = cls
        
        # Confidence smoothing
        self.conf_history = deque(maxlen=8)
        self.conf_history.append(conf)
        self.conf = conf

    def update(self, bbox, cls, conf):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.cls = cls
        
        # Exponential weighted moving average for confidence
        self.conf_history.append(conf)
        weights = [0.9 ** i for i in range(len(self.conf_history))]
        self.conf = sum(c * w for c, w in zip(reversed(self.conf_history), weights)) / sum(weights)

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x.flatten())

    def get_state(self):
        return convert_x_to_bbox(self.kf.x.flatten())


class BoTSORT:
    """BoT-SORT tracker with optimized parameters"""
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.trackers = []
        self.frame_count = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, dets=np.empty((0, 6))):
        self.frame_count += 1
        if len(dets) == 0:
            dets = np.empty((0, 6))

        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self.associate(dets[:, :4], trks)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4], int(dets[m[0], 5]), dets[m[0], 4])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4], int(dets[i, 5]), dets[i, 4]))

        ret = {}
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret[trk.id] = (d, trk.conf, trk.cls)
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return ret

    def associate(self, dets, trks):
        if len(trks) == 0:
            return np.empty((0, 2), int), np.arange(len(dets)), np.empty((0,), int)
        iou_matrix = iou_batch(dets, trks)
        matched = linear_assignment(-iou_matrix)
        unmatched_dets = [d for d in range(len(dets)) if d not in matched[:, 0]]
        unmatched_trks = [t for t in range(len(trks)) if t not in matched[:, 1]]
        matches = []
        for m in matched:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m)
        return np.array(matches), np.array(unmatched_dets), np.array(unmatched_trks)


class AlertManager:
    """Intelligent alert system with cooldown"""
    def __init__(self, cooldown_seconds=5):
        self.cooldown = cooldown_seconds
        self.last_alert = {}
        self.alert_count = 0
        
    def should_alert(self, track_id):
        current_time = time.time()
        if track_id not in self.last_alert:
            self.last_alert[track_id] = current_time
            self.alert_count += 1
            return True
        
        if current_time - self.last_alert[track_id] > self.cooldown:
            self.last_alert[track_id] = current_time
            self.alert_count += 1
            return True
        return False
    
    def cleanup(self, active_ids):
        stale = [tid for tid in self.last_alert if tid not in active_ids]
        for tid in stale:
            del self.last_alert[tid]


# ====================== CAMERA CONFIGURATION ======================
CAMERAS = [
    {"name": "CAM 1", "url": "rtsp://admin:pakistan%40123@10.115.50.163:554/ch1/main/av_stream"},
    {"name": "CAM 2", "url": "rtsp://admin:pakistan%40123@10.115.50.159:554/ch1/main/av_stream"},
    {"name": "CAM 3", "url": "rtsp://admin:pakistan%40123@10.115.50.161:554/ch1/main/av_stream"},
    {"name": "CAM 4", "url": "rtsp://admin:pakistan%40123@10.115.50.154:554/ch1/main/av_stream"},
    {"name": "CAM 5", "url": "rtsp://admin:pakistan%40123@10.115.50.162:554/ch1/main/av_stream"},
    {"name": "CAM 6", "url": "rtsp://admin:pakistan%40123@10.115.50.155:554/ch1/main/av_stream"},
    {"name": "CAM 7", "url": "rtsp://admin:pakistan%40123@10.115.50.158:554/ch1/main/av_stream"},
    {"name": "CAM 8", "url": "rtsp://admin:pakistan%40123@10.115.50.156:554/ch1/main/av_stream"},
    {"name": "CAM 9", "url": "rtsp://admin:pakistan%40123@10.115.50.167:554/ch1/main/av_stream"},
    {"name": "CAM 10", "url": "rtsp://admin:pakistan%40123@10.115.50.153:554/ch1/main/av_stream"},
    {"name": "CAM 11", "url": "rtsp://admin:pakistan%40123@10.115.50.157:554/ch1/main/av_stream"},
    {"name": "CAM 12", "url": "rtsp://admin:pakistan%40123@10.115.50.160:554/ch1/main/av_stream"}
]


# ====================== SESSION STATE INITIALIZATION ======================
if 'detection_running' not in st.session_state:
    st.session_state.detection_running = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'fps': 0.0,
        'detections': 0,
        'tracks': 0,
        'alerts': 0
    }
if 'detection_thread' not in st.session_state:
    st.session_state.detection_thread = None


# ====================== DETECTION FUNCTION ======================
def run_detection(config):
    """Run detection in background thread"""
    try:
        # Initialize capture
        src = config['source']
        if str(src).isdigit():
            src = int(src)
        
        cap = cv2.VideoCapture(src)
        if "rtsp" in str(src).lower():
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            st.session_state.detection_running = False
            return

        # Load model
        from ultralytics import YOLO
        model = YOLO(config['weights'])
        if torch.cuda.is_available():
            model.to('cuda')

        # Initialize components
        tracker = BoTSORT(
            max_age=config['max_age'],
            min_hits=config['min_hits'],
            iou_threshold=config['iou_threshold']
        )
        confirmed_tracker = ConfirmedDetectionTracker(config['confirmation_frames'])
        scene_filter = SceneContextFilter(config['edge_margin'])
        alert_manager = AlertManager(config['alert_cooldown'])

        # Video writer
        out = None
        if config['save']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            w = int(cap.get(3)) or 1280
            h = int(cap.get(4)) or 720
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = cv2.VideoWriter(f"output_{timestamp}.mp4", fourcc, 20, (w, h))

        prev_time = time.time()
        fps_counter = 0

        while st.session_state.detection_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if "rtsp" in str(config['source']).lower():
                    time.sleep(2)
                    cap.release()
                    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                    continue
                else:
                    break

            # Run detection
            results = model(
                frame,
                conf=config['conf'],
                iou=config['nms_iou'],
                imgsz=640,
                verbose=False
            )[0]

            # Extract detections
            dets = []
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf.cpu().item()
                    cls = int(box.cls.cpu().item())
                    dets.append([x1, y1, x2, y2, conf, cls])
            dets = np.array(dets) if dets else np.empty((0, 6))

            raw_det_count = len(dets)

            # Apply filters
            if config['use_size_filter']:
                dets = filter_by_size(dets, frame.shape, 
                                    config['min_size_ratio'],
                                    config['max_size_ratio'])
            
            if config['use_aspect_filter']:
                dets = validate_aspect_ratio(dets,
                                            config['min_aspect_ratio'],
                                            config['max_aspect_ratio'])
            
            if config['use_edge_filter']:
                dets = scene_filter.filter_edge_detections(dets, frame.shape)

            # Update tracker
            tracked = tracker.update(dets)

            # Apply temporal confirmation
            confirmed_tracks = {}
            active_ids = set()
            for tid, (bbox, conf, cls) in tracked.items():
                active_ids.add(tid)
                confirmed_tracker.add_detection(tid, cls)
                
                if config['use_temporal_filter']:
                    if confirmed_tracker.is_confirmed(tid):
                        confirmed_tracks[tid] = (bbox, conf, cls)
                else:
                    confirmed_tracks[tid] = (bbox, conf, cls)

            # Cleanup
            confirmed_tracker.cleanup(active_ids)
            alert_manager.cleanup(active_ids)

            # Draw detections
            for tid, (bbox, conf, cls) in confirmed_tracks.items():
                x1, y1, x2, y2 = map(int, bbox)
                
                # Color based on confidence
                if conf > 0.7:
                    color = (0, 0, 255)  # Red - High confidence
                elif conf > 0.5:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 255)  # Yellow - Low confidence
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Alert check
                if alert_manager.should_alert(tid) and conf > 0.6:
                    cv2.putText(frame, "ALERT!", (x1, y1 - 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                label = f"ID:{tid} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save frame
            if out:
                out.write(frame)

            # FPS calculation
            fps_counter += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = fps_counter / (current_time - prev_time)
                st.session_state.stats['fps'] = fps
                fps_counter = 0
                prev_time = current_time

            # Update stats and frame
            st.session_state.stats['detections'] = raw_det_count
            st.session_state.stats['tracks'] = len(confirmed_tracks)
            st.session_state.stats['alerts'] = alert_manager.alert_count
            st.session_state.current_frame = frame

        cap.release()
        if out:
            out.release()

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.detection_running = False


# ====================== STREAMLIT UI ======================
def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>🛡️ Enhanced Weapon Detection System v4.0</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #95a5a6;'>Advanced AI-powered weapon detection with temporal filtering and tracking</p>", 
                unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar - Controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Configuration
        st.subheader("🎯 Model Settings")
        model_path = st.text_input("Model Path (.pt)", "train15/weights/best.pt")
        
        # Video Source
        st.subheader("📷 Video Source")
        source_type = st.selectbox(
            "Source Type",
            ["Webcam", "Video File", "RTSP Stream", "Camera List"]
        )
        
        source = "0"
        if source_type == "Webcam":
            source = "0"
        elif source_type == "Video File":
            uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
            if uploaded_file:
                source = uploaded_file.name
        elif source_type == "RTSP Stream":
            source = st.text_input("RTSP URL", "rtsp://")
        elif source_type == "Camera List":
            camera_names = [cam["name"] for cam in CAMERAS]
            selected_camera = st.selectbox("Select Camera", camera_names)
            camera_idx = camera_names.index(selected_camera)
            source = CAMERAS[camera_idx]["url"]
            st.info(f"📡 {CAMERAS[camera_idx]['url']}")
        
        # Detection Parameters
        st.subheader("🎚️ Detection Parameters")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
        iou_threshold = st.slider("IOU Threshold", 0.2, 0.8, 0.45, 0.05)
        
        # False Alarm Reduction
        st.subheader("🛡️ False Alarm Reduction")
        use_temporal = st.checkbox("✓ Temporal Consistency Filter", value=True)
        confirmation_frames = st.slider("Confirmation Frames", 3, 10, 5)
        use_size = st.checkbox("✓ Size-Based Filter", value=True)
        use_aspect = st.checkbox("✓ Aspect Ratio Filter", value=True)
        use_edge = st.checkbox("✓ Edge Detection Filter", value=True)
        
        # Tracker Settings
        st.subheader("🎯 Tracker Settings")
        max_age = st.slider("Max Age (frames)", 10, 60, 30)
        min_hits = st.slider("Min Hits", 1, 10, 3)
        
        # Alert Settings
        st.subheader("🔔 Alert Settings")
        alert_cooldown = st.slider("Alert Cooldown (seconds)", 1, 30, 5)
        
        # Recording
        st.subheader("💾 Recording")
        save_video = st.checkbox("Save Video Output")
        
        st.markdown("---")
        
        # Control Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ START" if not st.session_state.detection_running else "🔄 RESTART", 
                        use_container_width=True):
                if st.session_state.detection_running:
                    st.session_state.detection_running = False
                    time.sleep(0.5)
                
                config = {
                    'source': source,
                    'weights': model_path,
                    'conf': conf_threshold,
                    'nms_iou': iou_threshold,
                    'save': save_video,
                    'confirmation_frames': confirmation_frames,
                    'max_age': max_age,
                    'min_hits': min_hits,
                    'iou_threshold': 0.3,
                    'use_temporal_filter': use_temporal,
                    'use_size_filter': use_size,
                    'use_aspect_filter': use_aspect,
                    'use_edge_filter': use_edge,
                    'min_size_ratio': 0.003,
                    'max_size_ratio': 0.4,
                    'min_aspect_ratio': 0.15,
                    'max_aspect_ratio': 6.0,
                    'edge_margin': 0.05,
                    'alert_cooldown': alert_cooldown
                }
                
                st.session_state.detection_running = True
                st.session_state.detection_thread = threading.Thread(
                    target=run_detection,
                    args=(config,),
                    daemon=True
                )
                st.session_state.detection_thread.start()
        
        with col2:
            if st.button("⏹️ STOP", use_container_width=True, type="secondary"):
                st.session_state.detection_running = False
                st.session_state.current_frame = None

    # Main Content
    col1, col2, col3, col4 = st.columns(4)
    
    fps_placeholder = col1.empty()
    det_placeholder = col2.empty()
    track_placeholder = col3.empty()
    alert_placeholder = col4.empty()
    
    # Video Feed
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()
    
    # Status
    status_placeholder = st.empty()
    
    # Continuous display loop
    if st.session_state.detection_running:
        status_placeholder.success("🟢 Detection Running")
        
        # Auto-refresh to show frames
        while st.session_state.detection_running:
            if st.session_state.current_frame is not None:
                frame_rgb = cv2.cvtColor(st.session_state.current_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update metrics
                fps_placeholder.metric("📊 FPS", f"{st.session_state.stats['fps']:.1f}")
                det_placeholder.metric("🔍 Raw Detections", st.session_state.stats['detections'])
                track_placeholder.metric("✅ Confirmed Tracks", st.session_state.stats['tracks'])
                alert_placeholder.metric("🚨 Total Alerts", st.session_state.stats['alerts'])
            
            time.sleep(0.03)  # ~30 FPS display rate
    else:
        status_placeholder.info("⏸️ Detection Stopped - Press START to begin")
        
        # Show placeholder image
        placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
        placeholder[:] = (26, 26, 46)  # Dark background
        
        # Add text
        text = "Press START to Begin Detection"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        text_x = (placeholder.shape[1] - text_size[0]) // 2
        text_y = (placeholder.shape[0] + text_size[1]) // 2
        cv2.putText(placeholder, text, (text_x, text_y), font, 1.5, (236, 240, 241), 3)
        
        video_placeholder.image(placeholder, channels="BGR", use_container_width=True)
        
        # Initialize metrics
        fps_placeholder.metric("📊 FPS", "0.0")
        det_placeholder.metric("🔍 Raw Detections", "0")
        track_placeholder.metric("✅ Confirmed Tracks", "0")
        alert_placeholder.metric("🚨 Total Alerts", "0")


if __name__ == "__main__":
    main()