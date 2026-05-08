<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

# SIGNLINGO

<em>Break Barriers, Speak Sign, Empower Everyone</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/last-commit/RicardoUrbaez/SignLingo?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/RicardoUrbaez/SignLingo?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/RicardoUrbaez/SignLingo?style=flat&color=0080ff" alt="repo-language-count">

<br>
<br>

<em>Built with the tools and technologies:</em>

<br>
<br>

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Flask-000000.svg?style=flat&logo=Flask&logoColor=white" alt="Flask">
<img src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=flat&logo=OpenCV&logoColor=white" alt="OpenCV">
<img src="https://img.shields.io/badge/MediaPipe-0097A7.svg?style=flat&logo=MediaPipe&logoColor=white" alt="MediaPipe">
<img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/MongoDB-47A248.svg?style=flat&logo=MongoDB&logoColor=white" alt="MongoDB">
<img src="https://img.shields.io/badge/Auth0-EB5424.svg?style=flat&logo=Auth0&logoColor=white" alt="Auth0">
<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">

</div>

<br>

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Reference Images](#reference-images)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Dataset and Model Training](#dataset-and-model-training)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Training the Model](#training-the-model)
- [Team](#team)
- [Notes](#notes)
- [Future Improvements](#future-improvements)
- [Documentation](#documentation)
- [License](#license)

---

## Overview

SignLingo is a 2026 capstone project created by StackFive.

SignLingo is a real-time computer vision sign language translator that uses webcam input to detect hand gestures, classify ASL signs, and translate them into readable English output. The project is intended for accessibility, education, and sign language learning.

The app combines a Flask web interface with a Python-based computer vision pipeline. OpenCV handles webcam capture and image processing, MediaPipe tracks hand landmarks from live video frames, and a scikit-learn model predicts the sign being shown.

The purpose of SignLingo is to reduce communication barriers between sign language users and non-signers by translating gestures into English output in real time. It also supports sign language learning through instant on-screen feedback and optional speech output.

This project was adapted and modified for academic use. Some starter computer vision structure was inspired by open-source sign language recognition examples.

---

## Demo

A short demo video is included to show the SignLingo interface, live camera feed, ASL reference panel, and recognition workflow.

https://github.com/user-attachments/assets/ab95413e-766e-43af-b9dd-b21a108880bf

<p align="center">
  <a href="docs/media/signlingo-demo.mp4">
    <strong>▶ Download the SignLingo Demo Video</strong>
  </a>
</p>
---

## Features

- Real-time hand gesture recognition from webcam input
- ASL sign classification using a trained machine learning model
- Live text output for recognized letters, words, and short phrases
- Flask-based web interface for running the app in a browser
- Dataset collection and model training workflow included in the repository
- Browser-based text-to-speech support for detected signs
- Auth0 login and sign-up flow for protected dashboard access
- MongoDB Atlas support for prediction logging and database documentation
- Built for education, accessibility, and sign language practice

---

## Reference Images

The app includes ASL reference images to help users compare their hand signs with the expected alphabet and phrase gestures.

### ASL Alphabet Reference

<p align="center">
  <img src="https://raw.githubusercontent.com/RicardoUrbaez/SignLingo/main/docs/media/asl-alphabet-reference.png" alt="ASL Alphabet Reference" width="700">
</p>

### ASL Phrase Reference

<p align="center">
  <img src="https://raw.githubusercontent.com/RicardoUrbaez/SignLingo/main/docs/media/asl-phrase-reference.jpg" alt="ASL Phrase Reference" width="700">
</p>

---

## Tech Stack

- Python
- Flask
- OpenCV
- MediaPipe
- scikit-learn
- NumPy
- MongoDB Atlas
- Auth0
- HTML
- CSS
- JavaScript
- Jinja Templates

---

## Project Structure

```text
SignLingo-Capstone/
├── app/
│   ├── __init__.py
│   ├── app.py
│   └── inference_classifier.py
├── scripts/
│   ├── collect_imgs.py
│   ├── create_dataset.py
│   └── train_classifier.py
├── models/
│   ├── data.pickle
│   └── model.p
├── data/
│   ├── raw/
│   └── processed/
├── static/
│   ├── audio/
│   ├── css/
│   ├── images/
│   └── js/
├── templates/
├── docs/
│   ├── media/
│   │   ├── signlingo-demo.mp4
│   │   ├── asl-alphabet-reference.png
│   │   └── asl-phrase-reference.jpg
│   ├── overview.md
│   └── database_schema.md
└── archive/
    └── needs_review/
```

- `app/` contains the Flask app and standalone inference entry point
- `scripts/` contains dataset collection, dataset building, and model training utilities
- `models/` stores the generated dataset pickle and trained classifier
- `data/raw/` stores collected class images and `data/processed/` is reserved for future processed outputs
- `static/` stores images, CSS, JavaScript, and audio assets used by the live app
- `templates/` stores the Flask/Jinja HTML templates
- `docs/media/` stores README media such as the demo video and ASL reference images
- `docs/` stores supporting project documentation
- `archive/needs_review/` stores legacy or uncertain files that were intentionally kept out of the main app structure

---

## How It Works

1. The webcam captures live video frames.
2. OpenCV processes the camera feed.
3. MediaPipe detects the hand and extracts landmark positions.
4. The landmark data is passed to a trained scikit-learn classifier.
5. The model predicts the sign being shown.
6. The predicted result is displayed in the web interface as text.
7. If enabled, the text-to-speech feature reads the detected sign aloud.
8. MongoDB can store prediction history and confidence data for later review.

The model is trained on collected hand sign images stored in the repository dataset. Those images are processed into landmark-based training data before the classifier is trained.

The main application and scripts resolve paths from their own file locations, so moving code into folders does not break model, template, static, or dataset loading.

---

## Dataset and Model Training

The project uses a custom dataset built from ASL hand sign images.

- 32 total sign classes
- 26 alphabet letters
- 6 common phrase signs
- 850 images per class
- Over 27,000 total training images

The dataset was collected using a webcam-based image collection workflow. Images were captured with variation in distance, angle, position, and hand orientation to improve model reliability.

---

## Installation

Build SignLingo from the source and install dependencies.

1. Clone the repository:

```bash
git clone https://github.com/RicardoUrbaez/SignLingo
```

2. Navigate to the project directory:

```bash
cd SignLingo
```

3. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

4. Install the project dependencies:

```bash
pip install -r requirements.txt
```

5. Add a `.env` file in the project root if your local setup requires MongoDB or Auth0 configuration.

Example:

```env
MONGODB_URI=your_mongodb_connection_string
MONGODB_DB=signlingo_db

AUTH0_DOMAIN=your_auth0_domain
AUTH0_CLIENT_ID=your_auth0_client_id
AUTH0_CLIENT_SECRET=your_auth0_client_secret
AUTH0_CALLBACK_URL=http://127.0.0.1:5000/callback
AUTH0_LOGOUT_URL=http://127.0.0.1:5000/signin
APP_SECRET_KEY=your_flask_secret_key
```

6. Make sure a webcam is connected and available.

---

## Running the App

Start the Flask app:

```bash
python app/app.py
```

Then open the browser at:

```text
http://127.0.0.1:5000/
```

If using the root-level run scripts:

```powershell
.\run.ps1
```

or:

```bat
run.bat
```

---

## Training the Model

1. Collect hand sign images:

```bash
python scripts/collect_imgs.py
```

2. Build the dataset file from the collected images:

```bash
python scripts/create_dataset.py
```

3. Train the classifier:

```bash
python scripts/train_classifier.py
```

4. Run the app again to use the updated model.

---

## Team

SignLingo was created by StackFive.

- Ricardo Alexander Urbaez — Lead AI Engineer
- Felix Molina — Co-Lead AI Engineer
- Michael Niemeyer — Database Administrator
- Nicole Bencosme-Gil — Front End Developer
- Naomi Joa — Front End Developer

---

## Team Contributions

Ricardo Alexander Urbaez led the AI integration work, including the computer vision pipeline, dataset collection, model training, live inference, and backend integration.

Felix Molina supported the AI integration and backend structure, including code refactoring, Auth0 authentication support, and system integration.

Michael Niemeyer contributed to database administration, including MongoDB Atlas setup, database schema planning, prediction logging structure, and data organization.

Nicole Bencosme-Gil contributed to the front-end interface, visual design, layout structure, and user experience.

Naomi Joa contributed to the front-end interface, user experience, design refinement, and presentation of the application.

---

## Notes

- Auth0 and MongoDB settings are loaded from `.env` at the project root.
- The `.env` file should not be committed to GitHub.
- Legacy helper files and inherited backups were moved into `archive/needs_review/` instead of being deleted.
- If you retrain the classifier, the updated artifacts are written back into `models/`.
- If the camera feed does not load, close other apps that may be using the webcam, such as Zoom, Teams, Discord, or the Windows Camera app.
- Large media files, such as demo videos, should be kept in `docs/media/` or uploaded through GitHub Releases if the file becomes too large for normal repository use.

---

## Future Improvements

- Expand support for more signs, words, and short phrases
- Improve recognition accuracy across different lighting and background conditions
- Add better feedback for uncertain predictions
- Support more structured learning and practice features
- Extend the system to cover a wider range of sign language lessons
- Add user progress tracking and practice history
- Build a dashboard for reviewing prediction logs and model performance
- Improve mobile and tablet support

---

## Documentation

- [Project overview](docs/overview.md)
- [Database schema](docs/database_schema.md)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for the full text.

---

<div align="left"><a href="#top">⬆ Return</a></div>
