import cv2
import torch
from ultralytics import YOLO
import argparse
import os
from pathlib import Path

def main():
    print("YOLO Inference Tool")
    print("-" * 50)
    
    # --- Model Selection ---
    model_path = input("Enter model path (.pt/.engine/.onnx): ").strip()
    if not model_path or not os.path.exists(model_path):
        print("Invalid model path!")
        return
    
    # --- Device ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")

    # --- Load Model ---
    print("Loading model...")
    model = YOLO(model_path)  # Supports .pt, .engine, .onnx
    model.to(device)
    
    # --- Inference Parameters ---
    conf = float(input("Confidence threshold [0.25]: ") or 0.25)
    iou = float(input("IoU threshold [0.45]: ") or 0.45)
    imgsz = int(input("Image size [640]: ") or 640)
    classes = input("Class IDs (comma-separated, e.g., 0,1 or blank): ").strip()
    classes = [int(c) for c in classes.split(',')] if classes else None

    # --- Source Selection ---
    print("\nSource options:")
    print("1. Image")
    print("2. Video file")
    print("3. Webcam")
    print("4. RTSP / CCTV")
    print("5. URL (image/video stream)")
    choice = input("Choose (1-5): ").strip()

    if choice == '1':
        source = input("Image path: ").strip()
        save = False
    elif choice == '2':
        source = input("Video path: ").strip()
        save = input("Save output? (y/n): ").lower() == 'y'
    elif choice == '3':
        source = 0
        save = input("Record webcam? (y/n): ").lower() == 'y'
    elif choice in ['4', '5']:
        source = input("RTSP/URL: ").strip()
        save = input("Record stream? (y/n): ").lower() == 'y'
    else:
        print("Invalid choice!")
        return

    # --- Output Setup ---
    out_path = None
    fourcc = None
    out = None
    if save and choice in ['2', '3', '4', '5']:
        out_dir = "inference_output"
        os.makedirs(out_dir, exist_ok=True)
        name = os.path.basename(source) if isinstance(source, str) and '/' in source else 'live'
        out_path = os.path.join(out_dir, f"output_{name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # --- Window Size ---
    win_w = int(input("Window width [1280]: ") or 1280)
    win_h = int(input("Window height [720]: ") or 720)
    cv2.namedWindow("YOLO Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Inference", win_w, win_h)

    # --- Run Inference ---
    print(f"\nStarting inference on: {source}")
    print("Press 'q' to quit, 's' to save frame (image only)")

    cap = cv2.VideoCapture(source)
    if choice in ['2', '4', '5'] and not cap.isOpened():
        print("Cannot open source!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if save and out_path:
        out = cv2.VideoWriter(out_path, fourcc, fps, (w or 640, h or 480))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if choice in ['3', '4', '5']:
                print("Stream failed. Reconnecting...")
                cap = cv2.VideoCapture(source)
                continue
            break

        frame_count += 1
        if frame is None: continue

        # Inference
        results = model(frame, conf=conf, iou=iou, imgsz=imgsz, classes=classes, device=device, verbose=False)[0]

        # Draw
        annotated_frame = frame.copy()
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = f"{model.names[cls_id]} {conf_score:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Info
        info = f"FPS: {fps:.1f} | Frame: {frame_count} | Dets: {len(results.boxes)}"
        cv2.putText(annotated_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("YOLO Inference", annotated_frame)
        if save and out:
            out.write(annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and choice == '1':
            save_img = f"saved_frame_{frame_count}.jpg"
            cv2.imwrite(save_img, annotated_frame)
            print(f"Saved: {save_img}")

    cap.release()
    if out:
        out.release()
        print(f"Recording saved: {out_path}")
    cv2.destroyAllWindows()
    print("Inference stopped.")

if __name__ == "__main__":
    main()