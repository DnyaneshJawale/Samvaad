SAMVAAD – Real-Time Indian Sign Language Communication System

**An AI-powered assistive communication platform that translates Indian Sign Language into text and speech in real time.**

---

## Overview

SAMVAAD is a browser-based assistive communication system developed to help bridge the communication gap between individuals who use Indian Sign Language (ISL) and those unfamiliar with it.

The application recognizes hand gestures through a standard webcam and converts recognized signs into readable text with optional speech output. The system is designed with accessibility, responsiveness, and real-time interaction as its primary goals.

---

## Key Features

* Real-time Indian Sign Language recognition
* Browser-based webcam interaction
* Live transcript generation
* Speech synthesis for recognized text
* Responsive and accessibility-focused user interface
* Real-time AI inference
* Confidence-aware prediction pipeline
* Modular full-stack architecture
* Cloud-deployable application

---

## Live Demonstration

**Application**

https://samvaad-ecru.vercel.app/

---

## Technology Stack

### Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS

### Backend

* FastAPI
* Python

### Artificial Intelligence

* TensorFlow / Keras
* MediaPipe
* OpenCV
* NumPy

### Deployment

* Vercel
* Render
* GitHub

---

## How It Works

The system follows a real-time recognition pipeline:

1. Capture live webcam frames.
2. Detect hand landmarks.
3. Process landmark sequences.
4. Perform AI-based sign recognition.
5. Generate transcript output.
6. Convert recognized text into speech.

This pipeline is optimized for responsive browser-based interaction while maintaining a smooth user experience.

---

## Project Highlights

* Real-time gesture recognition
* Continuous transcript generation
* Browser-based interaction
* AI-assisted communication support
* Accessibility-oriented design
* Scalable frontend-backend architecture

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/DnyaneshJawale/Samvaad.git

cd Samvaad
```

### Install Dependencies

Frontend

```bash
npm install
```

Backend

```bash
pip install -r requirements.txt
```

### Start the Application

Frontend

```bash
npm run dev
```

Backend

```bash
uvicorn ai_server:app --reload
```

---

## Project Structure

```text
SAMVAAD/
├── app/
├── components/
├── lib/
├── public/
├── trained_model/
├── ai_server.py
├── package.json
├── requirements.txt
└── README.md
```

---

## Objectives

* Improve accessibility through AI-assisted communication.
* Demonstrate real-time sign language recognition.
* Build a scalable full-stack application.
* Provide an intuitive communication interface.
* Explore practical applications of computer vision and deep learning.

---

## Current Capabilities

* Live webcam-based recognition
* Real-time transcript generation
* Speech output
* Responsive interface
* Browser-based operation
* Cloud deployment

---

## Future Enhancements

* Expanded vocabulary support
* Improved sentence-level recognition
* Enhanced multilingual capabilities
* Mobile application support
* Offline inference
* Performance optimization
* Improved accessibility features

---

## Academic Context

This project was developed as part of a Bachelor of Engineering Capstone Project in Electronics and Telecommunication Engineering. It demonstrates the integration of computer vision, artificial intelligence, and modern web technologies to address a practical accessibility challenge.

---

## License

This repository is provided for academic, educational, and demonstration purposes.

---

## Acknowledgements

The development of SAMVAAD was inspired by the need for accessible communication technologies and the growing potential of artificial intelligence to create inclusive digital solutions.
