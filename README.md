# SignLingo

## Overview

SignLingo is a 2026 capstone project by Ricardo Urbaez. It is a real-time sign language interpretation app that uses webcam input to detect hand gestures, classify ASL signs, and translate them into words or short phrases. The project is intended for education, accessibility, and sign language learning.

The app combines a Flask web interface with a computer vision pipeline built in Python. MediaPipe is used to track hand landmarks from live video frames, OpenCV handles webcam capture and image processing, and a scikit-learn model predicts the sign being shown.

This project was adapted and modified for academic use. Some starter computer vision structure was inspired by open-source sign language recognition examples.

## Features

- Real-time hand gesture recognition from webcam input
- ASL sign classification using a trained machine learning model
- Live text output for recognized signs, words, or phrases
- Flask-based web interface for running the app in a browser
- Dataset collection and model training workflow included in the repository
- Built for education, accessibility, and sign language practice

## Tech Stack

- Python
- Flask
- OpenCV
- MediaPipe
- scikit-learn
- HTML
- CSS
- JavaScript

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
│   └── database_schema.md
└── archive/
	└── needs_review/
```

- `app/` contains the Flask app and standalone inference entry point
- `scripts/` contains dataset collection, dataset building, and model training utilities
- `models/` stores the generated dataset pickle and trained classifier
- `data/raw/` stores collected class images and `data/processed/` is reserved for future processed outputs
- `docs/` stores supporting project documentation
- `archive/needs_review/` stores legacy or uncertain files that were intentionally kept out of the main app structure

## How It Works

1. The webcam captures live video frames.
2. MediaPipe detects the hand and extracts landmark positions.
3. The landmark data is passed to a trained scikit-learn classifier.
4. The model predicts the sign being shown.
5. The predicted result is displayed in the web interface as text.

The model is trained on collected hand sign images stored in the repository dataset. Those images are processed into landmark-based training data before the classifier is trained.

The main application and scripts now resolve paths from their own file locations, so moving code into folders does not break model, template, static, or dataset loading.

## Installation

1. Create and activate a virtual environment.

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

2. Install the project dependencies.

```bash
pip install -r requirements.txt
```

3. Add a `.env` file in the project root if your local setup requires MongoDB or Auth0 configuration.

4. Make sure a webcam is connected and available.

## Running the App

Start the Flask app:

```bash
python app/app.py
```

Then open the browser at:

```text
http://127.0.0.1:5000/
```

## Training the Model

1. Collect hand sign images with:

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

## Notes

- Auth0 and MongoDB settings are still loaded from `.env` at the project root.
- Legacy helper files and inherited backups were moved into `archive/needs_review/` instead of being deleted.
- If you retrain the classifier, the updated artifacts are written back into `models/`.

## Future Improvements

- Expand support for more signs, words, and short phrases
- Improve recognition accuracy across different lighting and background conditions
- Add better feedback for uncertain predictions
- Support more structured learning and practice features
- Extend the system to cover a wider range of sign language lessons

## Author

Ricardo Urbaez

## Documentation

- [Project overview](docs/overview.md)
- [Database schema](docs/database_schema.md)

## License

This project is licensed under the MIT License. See the `LICENSE` file for the full text.
