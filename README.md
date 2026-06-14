# 🚦 AI Civic Monitoring System

An AI-powered surveillance system that automatically detects civic violations such as **Garbage Littering**, **Trespassing**, and **No-Parking Violations** using **YOLOv8**, **Computer Vision**, and **Real-Time Video Analytics**.

---

## 📌 Project Description

The AI Civic Monitoring System is designed to enhance public safety and civic discipline by continuously monitoring surveillance footage and identifying activities that violate civic regulations. The system uses deep learning-based object detection and video analysis techniques to generate alerts and evidence for detected violations.

By automating the monitoring process, the system reduces manual supervision efforts and enables smarter city management.

---

## 🎯 Objectives

* Detect civic violations automatically in real time.
* Improve public safety and cleanliness.
* Reduce manual surveillance efforts.
* Generate evidence and violation records.
* Provide a monitoring dashboard for authorities.

---

## ✨ Key Features

### 🗑 Garbage / Litter Detection

Detects littering activities and identifies garbage objects in monitored areas.

### 🚷 Trespassing Detection

Detects unauthorized entry into restricted or protected zones.

### 🚗 No-Parking Violation Detection

Monitors parking-restricted areas and identifies illegally parked vehicles.

### 📹 Real-Time Monitoring

Processes live webcam or CCTV video feeds.

### 📊 Dashboard Visualization

Displays violation statistics, alerts, and monitoring information.

### 📁 Evidence Storage

Automatically saves snapshots of detected violations for future reference.

---

## 🛠 Technologies Used

| Technology  | Purpose                  |
| ----------- | ------------------------ |
| Python      | Core Development         |
| OpenCV      | Image & Video Processing |
| YOLOv8      | Object Detection         |
| Ultralytics | YOLO Framework           |
| Flask       | Web Application Backend  |
| HTML/CSS    | Frontend Interface       |
| JavaScript  | Dashboard Functionality  |
| Chart.js    | Data Visualization       |

---

## 🧠 System Architecture

Input Video Stream
↓
Frame Extraction
↓
YOLOv8 Object Detection
↓
Violation Detection Module
↓
Alert Generation
↓
Evidence Storage
↓
Dashboard Monitoring

---

## 📂 Project Structure

AI_Civic_Monitoring/

├── app/

├── models/

├── datasets/

├── outputs/

├── evidence/

├── logs/

├── runs/

├── videos/

├── dashboard.py

├── main.py

├── train.py

├── predict.py

├── webcam_detect.py

├── trespassing_detect.py

├── noparking_detect.py

├── requirements.txt

└── README.md

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/harshagjain-2005/AI_Civic_Monitoring.git
cd AI_Civic_Monitoring
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train Model

```bash
python train.py
```

### Test Model

```bash
python predict.py
```

### Run Real-Time Detection

```bash
python webcam_detect.py
```

### Launch Dashboard

```bash
python dashboard.py
```

---

## 📊 Dataset Information

The system is trained using custom annotated datasets in YOLO format.

### Classes Used

#### Garbage Detection

* Can
* Plastic Bottle
* Plastic Bottle Cap
* Plastic Cover
* Thermocol
* Mask
* Lifesaver
* Wood

#### Trespassing Detection

* Person
* Restricted Zone

#### No-Parking Detection

* Car
* Motorcycle
* Vehicle

---

## 📈 Expected Outcomes

* Real-time civic violation detection.
* Automated monitoring of public spaces.
* Reduced dependency on manual surveillance.
* Digital evidence collection.
* Improved civic awareness and enforcement.

---

## 🔮 Future Enhancements

* Integration with Smart City Infrastructure.
* Multi-Camera Support.
* Cloud-Based Monitoring.
* Mobile Application Integration.
* SMS / Email Alert System.
* License Plate Recognition.
* AI-Based Analytics and Reporting.

---

## 👨‍💻 Team Members

* Harsha G Jain
* Akash B Madiwal
* Akshath Kumar L K
* Ajay Sharma

---

## 🎓 Academic Information

**Project Title:** AI Civic Monitoring System

**Domain:** Artificial Intelligence & Computer Vision

**Technology Stack:** YOLOv8, OpenCV, Flask, Python

**Project Type:** Major Project

---

## 📜 Conclusion

The AI Civic Monitoring System demonstrates the practical application of Artificial Intelligence and Computer Vision in creating smarter and safer public environments. By automatically detecting civic violations and generating actionable insights, the system contributes toward efficient urban management and responsible civic behavior.

---

⭐ If you found this project useful, consider giving it a star on GitHub.
