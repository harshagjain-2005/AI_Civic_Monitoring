# AI Civic Monitoring System

## Project Overview

The AI Civic Monitoring System is an intelligent surveillance-based solution that automatically detects civic rule violations using Artificial Intelligence and Computer Vision techniques. The system analyzes video streams in real time and identifies activities such as garbage littering, trespassing, and no-parking violations.

## Objective

The main objective of this project is to improve public safety and civic discipline by automatically monitoring public areas and generating alerts whenever a violation is detected.

## Features

* Real-time video surveillance
* Garbage/Litter Detection
* Trespassing Detection
* No-Parking Violation Detection
* YOLOv8-based object detection
* Evidence image generation
* Monitoring dashboard
* Automated violation logging

## Technologies Used

* Python
* OpenCV
* YOLOv8
* Ultralytics
* Flask
* HTML, CSS, JavaScript
* Chart.js

## Dataset Used

The model is trained using custom datasets containing images of:

* Garbage and litter objects
* Trespassing scenarios
* Vehicles in no-parking zones

The datasets were annotated in YOLO format and used for training the detection models.

## Project Structure

AI_Civic_Monitoring/

├── app/

├── models/

├── datasets/

├── outputs/

├── evidence/

├── logs/

├── videos/

├── dashboard.py

├── main.py

├── train.py

├── predict.py

├── webcam_detect.py

└── requirements.txt

## How to Run

### 1. Install Dependencies

pip install -r requirements.txt

### 2. Train the Model

python train.py

### 3. Test the Model

python predict.py

### 4. Run Real-Time Detection

python webcam_detect.py

### 5. Launch Dashboard

python dashboard.py

## Expected Output

* Detection of civic violations in real time
* Alert generation
* Evidence image storage
* Dashboard-based monitoring and statistics

## Future Enhancements

* Integration with CCTV networks
* Mobile notification system
* Cloud-based monitoring
* Multi-camera support
* Advanced analytics and reporting

## Team Members

* Harsha G Jain
* Akasha B Madiwal
* Akshath kumar LK
* ajay Sharma

## Conclusion

The AI Civic Monitoring System demonstrates how Artificial Intelligence can be utilized to automatically monitor public spaces and promote civic responsibility through real-time violation detection and reporting.
