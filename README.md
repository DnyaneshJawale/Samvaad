# SAMVAAD — Real-Time Indian Sign Language Communication System

**Bachelor of Engineering Capstone Project**  
Department of Electronics and Telecommunication Engineering  
Savitribai Phule Pune University

---

## Live Demo

Frontend: https://samvaad-ecru.vercel.app/  
Backend: https://samvaad-3c3o.onrender.com/

---

## Abstract

SAMVAAD is a real-time Indian Sign Language (ISL) communication system designed to bridge the communication gap between sign language users and non-sign language users. The system captures live video from a webcam, extracts hand landmarks using MediaPipe, processes the landmark sequence through a trained deep learning model, and converts the detected sign into text and speech output.

The project is implemented as a browser-based web application using Next.js, TypeScript, and Tailwind CSS for the frontend, with FastAPI and TensorFlow/Keras for backend inference. The recognition pipeline is designed to support continuous sign detection and stable transcript generation in real time. The system is intended as an accessible communication aid and a deployable capstone-level AI application.

---

## 1. Project Overview

SAMVAAD is an AI-based assistive communication platform that recognizes Indian Sign Language gestures from a live camera feed and converts them into readable text. The application is designed for real-time interaction, with the goal of making communication smoother for deaf and mute users in practical day-to-day scenarios.

Unlike a simple gesture demo, this project focuses on the complete pipeline:
- webcam capture,
- landmark extraction,
- temporal sign recognition,
- transcript generation,
- speech synthesis,
- and deployable web-based interaction.

The project demonstrates the use of computer vision, temporal deep learning, and full-stack web engineering in a real-world accessibility use case.

---

## 2. Problem Statement

People who rely on sign language often face communication barriers when interacting with users unfamiliar with ISL. Conventional communication systems are either manual, slow, or not practical for real-time use. SAMVAAD addresses this problem by providing a real-time translation interface that can convert recognized signs into text and speech through a standard webcam and browser.

---

## 3. Objectives

The main objectives of the project are:

1. To capture live hand gestures using a webcam.
2. To extract hand landmarks in real time using MediaPipe.
3. To use a deep learning model for ISL sign classification.
4. To generate readable transcript output from recognized signs.
5. To produce speech output for the recognized text.
6. To build a deployable, browser-based, accessibility-first application.
7. To maintain a stable and practical user experience suitable for real communication.

---

## 4. System Architecture

SAMVAAD follows a layered architecture:

```text
Webcam Input
   ↓
MediaPipe Hand Landmarker
   ↓
Landmark Feature Extraction
   ↓
Temporal Sequence Buffer
   ↓
TensorFlow/Keras Recognition Model
   ↓
Prediction and Confidence Scoring
   ↓
Transcript Engine
   ↓
Speech Synthesis
   ↓
User Interface Output
