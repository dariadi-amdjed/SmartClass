# SmartClass

## Overview

SmartClass is an AI-powered classroom management system that automates attendance tracking, improves communication between schools and parents, and enhances classroom safety through computer vision and hardware integration.

The system combines facial recognition, automated notifications, classroom management tools, and fire detection capabilities into a single platform designed to support teachers in daily classroom operations.

Developed by **Mohamed Amdjed Dariadi**, SmartClass was tested in a real secondary school environment in Algeria.

---

## Problem Statement

Many schools still rely on manual attendance systems, which can be time-consuming, error-prone, and difficult to monitor efficiently. Communication with parents is often delayed, and safety monitoring systems are usually separated from educational management tools.

SmartClass aims to provide an integrated solution that addresses these challenges through automation and artificial intelligence.

---

## Key Features

### Face Recognition Attendance

* Automatic student attendance registration
* Teacher authentication through facial recognition
* Real-time attendance updates
* Reduced manual administrative work

### Parent Communication

* Automated WhatsApp notifications
* Absence alerts
* Attendance status updates
* Direct parent-school communication support

### Classroom Management

* Student registration and management
* Attendance record storage
* Teacher dashboard interface
* Local data management

### Safety Monitoring

* Fire detection using sensors
* Arduino-based hardware integration
* Emergency alerts
* Automated response mechanisms

---

## Technologies Used

* Python
* OpenCV
* face_recognition (dlib-based)
* Tkinter
* Arduino
* PyFirmata
* JSON Database Storage
* WhatsApp Web Integration

---

## System Architecture

```text
Camera
   ↓
Face Recognition Engine
   ↓
Attendance Processing
   ↓
Teacher Dashboard
   ↓
Parent Notifications

Fire Sensor
   ↓
Arduino Controller
   ↓
Safety Alert System
```

---

## Project Structure

```text
SmartClass/
│
├── teacher_app.py
├── firebase.py
├── fire_sensor.py
├── requirements.txt
│
└── data/
    ├── students.json
    ├── teachers.json
    ├── class_state.json
    ├── sent_messages.json
    ├── students_img/
    └── teacher_faces/
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/SmartClass.git
cd SmartClass
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data Files

Ensure the following folders and files exist:

```text
data/
├── students.json
├── teachers.json
├── class_state.json
├── sent_messages.json
├── students_img/
└── teacher_faces/
```

Store student images in `students_img/` and teacher images in `teacher_faces/`.

### 5. Optional Hardware Setup

To use fire detection features:

* Connect an Arduino board running Firmata.
* Update the serial port if needed.
* Connect the flame sensor and output components according to your hardware configuration.

### 6. Run the Application

```bash
python teacher_app.py
```

---

## Real-World Testing

SmartClass was tested in a real secondary school environment in Médéa, Algeria, to evaluate attendance automation and classroom management workflows.

The project demonstrated successful attendance tracking, parent notification functionality, and hardware integration capabilities.

---

## Future Improvements

* Cloud database integration
* Mobile application support
* Advanced analytics dashboard
* Multi-classroom management
* Improved face recognition accuracy
* Web-based monitoring platform

---

## Educational Purpose

SmartClass was developed as a student-led artificial intelligence and educational technology project to explore practical applications of computer vision, automation, and embedded systems in schools.

---
