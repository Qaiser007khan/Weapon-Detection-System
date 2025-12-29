🔫 Weapon Detection System for Intelligent Surveillance

Real-Time Weapon Detection for Proactive Surveillance Using Deep Learning and Computer Vision

This repository presents a real-time weapon detection system designed for intelligent video surveillance, capable of detecting handguns, rifles, knives, and other weapons in live video streams and recorded footage.

The system leverages state-of-the-art object detection models to balance accuracy, speed, and reliability, enabling early threat detection in public and private environments.

🚨 Why Weapon Detection?

Traditional surveillance systems are passive—they record incidents but do not prevent them.
This system transforms surveillance into a proactive safety tool by automatically detecting weapons and enabling real-time alerts.

Key Use Cases

🏫 Schools & Universities

🏢 Corporate Offices

🛍️ Shopping Malls

🚉 Airports & Transport Hubs

🏙️ Smart City Surveillance

🎯 System Highlights

✅ Real-time weapon detection (guns, knives, rifles)

✅ Works on CCTV, RTSP streams, and video files

✅ Optimized for low latency surveillance

✅ Scalable to multi-camera setups

✅ Edge & GPU deployment ready

🚀 Demo Results
🔹 Weapon Detection on Images
<p align="center"> <img src="demo/images/gun_1.jpg" width="420"/> <img src="demo/images/knife_1.jpg" width="420"/> </p> <p align="center"> <img src="demo/images/gun_2.jpg" width="420"/> <img src="demo/images/rifle_1.jpg" width="420"/> </p>
🔹 Weapon Detection on Video (Real-Time)

🎥 Live Surveillance Demo

<p align="center"> <a href="demo/video/weapon_detection_demo.mp4"> <img src="demo/video/video_thumbnail.jpg" width="720"/> </a> </p>

The system detects weapons frame-by-frame and can be integrated with alarms, dashboards, or security control rooms.

🧠 Model Architecture & Approach

The system is built on modern object detection architectures, evaluated for surveillance-specific requirements such as:

Small object detection

Occlusion handling

Low false-alarm rate

Real-time inference

Models Evaluated

YOLOv8

YOLOv9

YOLOv10

RT-DETR

📊 Model Performance Comparison
🔍 Accuracy vs Speed Trade-off
Model	Precision	Recall	mAP@50	Inference Time
RT-DETR	0.933	0.561	0.672	❌ 26.5 ms
YOLOv8m	0.545	0.577	0.582	✅ 7.0 ms
YOLOv9m	0.612	0.541	0.573	11.2 ms
YOLOv10m	0.975	0.308	0.554	9.1 ms
YOLOv10l	0.821	0.486	0.569	⚡ 9.0 ms
🏆 Final Verdict

🎯 Highest Accuracy → RT-DETR (best for forensic analysis)

⚡ Best Real-Time Performance → YOLOv10l

⚖️ Best Speed–Accuracy Balance → YOLOv8m

🗂️ Dataset Overview

Curated weapon datasets with:

Handguns

Knives

Rifles

Real-world CCTV-like scenarios

Diverse lighting, viewpoints, and occlusions

Balanced negative samples to reduce false alarms

📁 Dataset Structure
dataset/
 ├── images/
 │   ├── train
 │   ├── val
 │   └── test
 └── labels/
     ├── train
     ├── val
     └── test

⚙️ Environment Setup
pip install -r requirements.txt

Tested Configuration

Python 3.9

PyTorch 2.x

CUDA 11.x

YOLOv8 / YOLOv10

GPU: RTX / Tesla / Jetson

🏋️ Training
yolo task=detect mode=train model=yolov8m.yaml data=weapon.yaml epochs=100 imgsz=640

🔍 Inference
Image
yolo task=detect mode=predict model=best.pt source=demo/images

Video / CCTV
yolo task=detect mode=predict model=best.pt source=rtsp://camera_ip

🚦 System Integration

This weapon detection system can be integrated with:

🚨 Alarm & siren systems

📡 Control room dashboards

🧠 AI-based threat analysis

🎯 Access control systems

🛡️ Ethical Considerations

Designed strictly for safety and security

No facial recognition or identity inference

Focused on object-based threat detection

Supports privacy-preserving deployments

📌 Applications

Smart surveillance systems

Automated threat detection

Public safety monitoring

Industrial & corporate security

Smart city infrastructure

📖 Citation

If you use this work in your research or deployment, please cite:

@article{khan2025weapondetection,
  title={Real-Time Weapon Detection for Intelligent Surveillance Using Deep Learning},
  author={Khan, Qaiser},
  year={2025},
  journal={Under Review}
}

👨‍💻 Author

Qaiser Khan
AI & Robotics Engineer | Computer Vision Researcher
🔗 GitHub: https://github.com/Qaiser007khan
