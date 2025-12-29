import cv2
import torch
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import argparse
from collections import defaultdict
import streamlit as st
import tempfile
from PIL import Image


class PersonWeaponDetector:
    def __init__(self, person_model_path, weapon_model_path, output_dir='threat_detection', 
                 person_conf=0.5, weapon_conf=0.5):
        """
        Initialize the fused person-weapon detection system with tracking
        
        Args:
            person_model_path: Path to person detection model (e.g., 'yolov8n.pt')
            weapon_model_path: Path to weapon detection model
            output_dir: Directory to save threat detection screenshots
            person_conf: Confidence threshold for person detection (default: 0.5)
            weapon_conf: Confidence threshold for weapon detection (default: 0.5)
        """
        # Set device to GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load models
        self.person_model = YOLO(person_model_path)
        self.person_model.to(self.device)
        
        self.weapon_model = YOLO(weapon_model_path)
        self.weapon_model.to(self.device)
        
        # Set confidence thresholds
        self.person_conf = person_conf
        self.weapon_conf = weapon_conf
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
        
        # Tracking data structures
        self.tracked_threats = {}
        self.saved_threat_ids = set()
    
    def get_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0
        
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    def is_weapon_near_person(self, person_box, weapon_box, threshold=0.1):
        """Check if weapon is near or overlapping with person"""
        iou = self.get_iou(person_box, weapon_box)
        if iou > threshold:
            return True
        
        px1, py1, px2, py2 = person_box
        wx1, wy1, wx2, wy2 = weapon_box
        
        width = px2 - px1
        height = py2 - py1
        expanded_box = [
            px1 - width * 0.2,
            py1 - height * 0.2,
            px2 + width * 0.2,
            py2 + height * 0.2
        ]
        
        weapon_center_x = (wx1 + wx2) / 2
        weapon_center_y = (wy1 + wy2) / 2
        
        if (expanded_box[0] <= weapon_center_x <= expanded_box[2] and
            expanded_box[1] <= weapon_center_y <= expanded_box[3]):
            return True
        
        return False
    
    def process_frame(self, frame, frame_number=0):
        """Process a single frame with person tracking and weapon detection"""
        person_results = self.person_model.track(frame, persist=True, verbose=False, conf=self.person_conf)[0]
        
        persons = []
        for box in person_results.boxes:
            if int(box.cls[0]) == self.person_class_id:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else None
                
                persons.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': conf,
                    'track_id': track_id,
                    'has_weapon': False
                })
        
        weapons = []
        threat_detected = False
        new_threat_ids = []
        
        if len(persons) > 0:
            weapon_results = self.weapon_model(frame, verbose=False, conf=self.weapon_conf)[0]
            
            for box in weapon_results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                weapon_box = [int(x1), int(y1), int(x2), int(y2)]
                
                for person in persons:
                    if self.is_weapon_near_person(person['box'], weapon_box):
                        person['has_weapon'] = True
                        threat_detected = True
                        
                        if person['track_id'] is not None:
                            track_id = person['track_id']
                            
                            if track_id not in self.tracked_threats:
                                self.tracked_threats[track_id] = {
                                    'first_seen': frame_number,
                                    'saved': False
                                }
                                new_threat_ids.append(track_id)
                            else:
                                self.tracked_threats[track_id]['last_seen'] = frame_number
                        
                        weapons.append({
                            'box': weapon_box,
                            'conf': conf
                        })
                        break
        
        output_frame = frame.copy()
        
        for person in persons:
            box = person['box']
            track_id = person['track_id']
            
            if person['has_weapon']:
                color = (0, 0, 255)
                if track_id is not None:
                    label = f"THREAT ID:{track_id} {person['conf']:.2f}"
                else:
                    label = f"Person + Weapon {person['conf']:.2f}"
                thickness = 3
            else:
                color = (0, 255, 0)
                if track_id is not None:
                    label = f"Person ID:{track_id} {person['conf']:.2f}"
                else:
                    label = f"Person {person['conf']:.2f}"
                thickness = 2
            
            cv2.rectangle(output_frame, (box[0], box[1]), (box[2], box[3]), color, thickness)
            
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                output_frame,
                (box[0], box[1] - label_height - 10),
                (box[0] + label_width, box[1]),
                color,
                -1
            )
            cv2.putText(
                output_frame, label,
                (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        for weapon in weapons:
            box = weapon['box']
            color = (0, 255, 255)
            label = f"Weapon {weapon['conf']:.2f}"
            
            cv2.rectangle(output_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(output_frame, label, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_frame, threat_detected, new_threat_ids
    
    def save_threat_screenshot(self, frame, threat_id):
        """Save screenshot of threat detection with unique ID"""
        if threat_id in self.saved_threat_ids:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"threat_ID{threat_id}_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        
        self.saved_threat_ids.add(threat_id)
        if threat_id in self.tracked_threats:
            self.tracked_threats[threat_id]['saved'] = True
        
        return filename


def streamlit_app():
    """Streamlit web interface for person-weapon detection"""
    st.set_page_config(page_title="Person-Weapon Detection System", layout="wide")
    
    st.title("🔍 Person-Weapon Detection System")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        person_model = st.text_input("Person Model Path", value="yolov8n.pt")
        weapon_model = st.text_input("Weapon Model Path", value="weapon_detection.pt")
        
        st.markdown("---")
        st.subheader("Detection Thresholds")
        person_conf = st.slider("Person Confidence", 0.0, 1.0, 0.5, 0.05)
        weapon_conf = st.slider("Weapon Confidence", 0.0, 1.0, 0.5, 0.05)
        
        st.markdown("---")
        output_dir = st.text_input("Output Directory", value="threat_detection")
        
        st.markdown("---")
        st.info(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📹 Webcam (Live)"])
    
    # Tab 1: Image Detection
    with tab1:
        st.header("Upload Image for Detection")
        
        uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp'], key="image")
        
        if uploaded_image is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_image)
                st.image(image, use_container_width=True)
            
            # Process button
            if st.button("🔍 Detect Threats", key="detect_image"):
                with st.spinner("Processing image..."):
                    try:
                        # Initialize detector
                        detector = PersonWeaponDetector(
                            person_model_path=person_model,
                            weapon_model_path=weapon_model,
                            output_dir=output_dir,
                            person_conf=person_conf,
                            weapon_conf=weapon_conf
                        )
                        
                        # Convert PIL to OpenCV format
                        img_array = np.array(image)
                        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        # Process frame
                        processed_frame, threat_detected, new_threat_ids = detector.process_frame(frame, 0)
                        
                        # Convert back to RGB for display
                        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display processed image
                        with col2:
                            st.subheader("Detection Result")
                            st.image(processed_rgb, use_container_width=True)
                        
                        # Show results
                        if threat_detected:
                            st.error(f"⚠️ THREAT DETECTED! Threat IDs: {new_threat_ids}")
                            
                            # Save threat screenshots
                            for threat_id in new_threat_ids:
                                saved_path = detector.save_threat_screenshot(processed_frame, threat_id)
                                if saved_path:
                                    st.success(f"🚨 Threat ID {threat_id} saved: {saved_path}")
                        else:
                            st.success("✅ No threats detected")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Tab 2: Video Detection
    with tab2:
        st.header("Upload Video for Detection")
        
        uploaded_video = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov', 'mkv'], key="video")
        
        if uploaded_video is not None:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            st.video(uploaded_video)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                process_every_n = st.number_input("Process every N frames", min_value=1, max_value=30, value=5)
            with col2:
                save_output = st.checkbox("Save Processed Video", value=False)
            with col3:
                max_frames = st.number_input("Max frames to process (0=all)", min_value=0, value=0)
            
            if st.button("🔍 Process Video", key="detect_video"):
                with st.spinner("Processing video... This may take a while."):
                    try:
                        # Initialize detector
                        detector = PersonWeaponDetector(
                            person_model_path=person_model,
                            weapon_model_path=weapon_model,
                            output_dir=output_dir,
                            person_conf=person_conf,
                            weapon_conf=weapon_conf
                        )
                        
                        # Open video
                        cap = cv2.VideoCapture(video_path)
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Setup progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        frame_placeholder = st.empty()
                        
                        # Video writer if saving
                        video_writer = None
                        if save_output:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            output_video_path = Path(output_dir) / f"processed_{uploaded_video.name}"
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
                        
                        frame_count = 0
                        threat_count = 0
                        unique_threats = set()
                        
                        frames_to_process = max_frames if max_frames > 0 else total_frames
                        
                        while cap.isOpened() and frame_count < frames_to_process:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame_count += 1
                            
                            # Process every Nth frame
                            if frame_count % process_every_n == 0:
                                processed_frame, threat_detected, new_threat_ids = detector.process_frame(frame, frame_count)
                                
                                # Save new threats
                                for threat_id in new_threat_ids:
                                    if threat_id not in detector.saved_threat_ids:
                                        detector.save_threat_screenshot(processed_frame, threat_id)
                                        unique_threats.add(threat_id)
                                
                                if threat_detected:
                                    threat_count += 1
                                
                                # Display frame in Streamlit
                                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(rgb_frame, use_container_width=True)
                                
                                if video_writer is not None:
                                    video_writer.write(processed_frame)
                            
                            # Update progress
                            progress = frame_count / frames_to_process
                            progress_bar.progress(progress)
                            status_text.text(f"Processing: Frame {frame_count}/{frames_to_process} | Threats: {len(unique_threats)}")
                        
                        cap.release()
                        if video_writer is not None:
                            video_writer.release()
                        
                        # Show summary
                        st.markdown("---")
                        st.subheader("📊 Processing Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Frames", frame_count)
                        with col2:
                            st.metric("Frames with Threats", threat_count)
                        with col3:
                            st.metric("Unique Threats", len(unique_threats))
                        with col4:
                            detection_rate = (threat_count/frame_count*100) if frame_count > 0 else 0
                            st.metric("Detection Rate", f"{detection_rate:.2f}%")
                        
                        if len(unique_threats) > 0:
                            st.warning(f"⚠️ Threat IDs detected: {sorted(list(unique_threats))}")
                        
                        if save_output:
                            st.success(f"✅ Processed video saved to: {output_video_path}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                    finally:
                        # Cleanup
                        if os.path.exists(video_path):
                            os.unlink(video_path)
    
    # Tab 3: Webcam
    with tab3:
        st.header("Live Webcam Detection")
        st.info("⚠️ Webcam feature requires running the script locally with OpenCV display enabled.")
        st.markdown("""
        To use webcam detection, run the script from command line:
        ```bash
        python script.py --source 0 --person-conf 0.5 --weapon-conf 0.5
        ```
        """)
        
        webcam_index = st.number_input("Webcam Index", min_value=0, max_value=5, value=0)
        
        if st.button("📝 Generate Command", key="webcam_cmd"):
            cmd = f"python script.py --source {webcam_index} --person-conf {person_conf} --weapon-conf {weapon_conf}"
            st.code(cmd, language="bash")
            st.info("Copy and run this command in your terminal")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Fused Person-Weapon Detection System with Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Streamlit Web Interface
  streamlit run script.py
  
  # Webcam with custom weapon confidence
  python script.py --source 0 --weapon-conf 0.7
  
  # RTSP Stream
  python script.py --source rtsp://192.168.1.100:554/stream --weapon-conf 0.6
  
  # Video File
  python script.py --source video.mp4 --person-conf 0.6 --weapon-conf 0.75 --save-output
        """
    )
    
    parser.add_argument(
        '--streamlit',
        action='store_true',
        help='Run Streamlit web interface (default if no source specified)'
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default=None,
        help='Input source: webcam index (0, 1), RTSP URL, video file path, or image file path'
    )
    
    parser.add_argument(
        '--person-model', '-p',
        type=str,
        default='yolov8n.pt',
        help='Path to person detection model (default: yolov8n.pt)'
    )
    
    parser.add_argument(
        '--weapon-model', '-w',
        type=str,
        default='train18/weights/best.pt',
        help='Path to weapon detection model (default: weapon_detection.pt)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='threat_detection',
        help='Directory to save threat detection screenshots (default: threat_detection)'
    )
    
    parser.add_argument(
        '--person-conf',
        type=float,
        default=0.5,
        help='Confidence threshold for person detection (default: 0.5)'
    )
    
    parser.add_argument(
        '--weapon-conf',
        type=float,
        default=0.7,
        help='Confidence threshold for weapon detection (default: 0.5)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display video output (useful for headless systems)'
    )
    
    parser.add_argument(
        '--save-output',
        action='store_true',
        help='Save processed video output (only for video files)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # If no source specified or --streamlit flag, run Streamlit
    if args.source is None or args.streamlit:
        streamlit_app()
    else:
        # Command-line mode (original functionality)
        print("\n" + "="*60)
        print("PERSON-WEAPON DETECTION SYSTEM - COMMAND LINE MODE")
        print("="*60)
        print("For web interface, run: streamlit run script.py")
        print("="*60 + "\n")
        
        st.error("Command-line mode requires the full original code.")
        st.info("Use Streamlit interface above or implement CLI mode separately.")