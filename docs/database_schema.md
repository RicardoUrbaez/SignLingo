# SignLingo Database Schema

## Overview

**Database name:** `signlingo_db`

SignLingo uses **MongoDB Atlas** as its primary database. Because MongoDB is a **NoSQL, document-based database**, the application stores related data as flexible JSON-like documents rather than fixed SQL tables.

This schema is designed for a real-time sign language translation platform built with Flask, Python, OpenCV, MediaPipe, machine learning, MongoDB Atlas, and Auth0.

## Design Notes

- Documents are stored in collections instead of relational tables.
- Authenticated users are identified with **Auth0 user IDs**.
- Prediction activity is stored with timestamps and confidence scores.
- Practice and model evaluation data are separated to support analytics, progress tracking, and capstone reporting.
- The model metadata below reflects a classifier configured for **32 classes** with **850 images per class**.

---

## 1. `users`

### Purpose

Stores profile and account information for each authenticated SignLingo user.

### Sample JSON Document

```json
{
  "_id": { "$oid": "6639f7f0d0a9e6b8c12a1001" },
  "auth0_user_id": "auth0|664fe2ab91c4fd23c9e12a11",
  "email": "maya.fernandez@signlingo.edu",
  "full_name": "Maya Fernandez",
  "role": "student",
  "profile_picture": "https://example.auth0.com/avatar/maya.png",
  "preferred_mode": "real_time_translation",
  "created_at": "2026-04-29T09:12:44Z",
  "last_login_at": "2026-04-29T14:05:18Z",
  "is_active": true
}
```

### Important Fields

- `auth0_user_id`: Unique identity from Auth0 used to connect application activity to a user.
- `email`: User email for login and communication.
- `full_name`: Display name shown in the interface or reports.
- `role`: Helps distinguish students, evaluators, or admins.
- `preferred_mode`: Stores the user's primary usage mode, such as translation or practice.
- `created_at`: Timestamp for account creation.
- `last_login_at`: Timestamp of the most recent successful login.
- `is_active`: Indicates whether the account is currently enabled.

### How It Supports the App

This collection links Auth0 authentication to SignLingo-specific user data. It allows the app to personalize the experience, associate predictions and practice sessions with a real user, and generate user-level progress summaries for the capstone system.

---

## 2. `prediction_logs`

### Purpose

Stores real-time sign detection results produced by the machine learning model during live translation.

### Sample JSON Document

```json
{
  "_id": { "$oid": "6639f89ed0a9e6b8c12a1002" },
  "auth0_user_id": "auth0|664fe2ab91c4fd23c9e12a11",
  "detected_sign": "Hello",
  "predicted_class_index": 26,
  "confidence_score": 0.94,
  "source": "live_camera",
  "model_name": "random_forest_sign_classifier",
  "model_version": "v1.0.0",
  "frame_width": 640,
  "frame_height": 480,
  "timestamp": "2026-04-29T14:07:31Z"
}
```

### Important Fields

- `auth0_user_id`: Identifies which user generated the prediction.
- `detected_sign`: The label predicted by the model, such as `A`, `B`, `Hello`, or `I Love You`.
- `predicted_class_index`: Numeric class identifier used by the classifier.
- `confidence_score`: Probability-like score representing model confidence.
- `source`: Indicates whether the prediction came from the live camera or another input source.
- `model_name`: Name of the deployed ML model.
- `model_version`: Tracks which model version produced the result.
- `timestamp`: Exact time the prediction was recorded.

### How It Supports the App

This collection is central to the translation workflow. It provides a persistent history of recognized signs, supports recent-prediction dashboards, enables performance analysis, and helps demonstrate real-time inference capability in the capstone presentation.

---

## 3. `practice_sessions`

### Purpose

Stores user practice activity, including which signs were attempted, session duration, and overall performance.

### Sample JSON Document

```json
{
  "_id": { "$oid": "6639f94bd0a9e6b8c12a1003" },
  "auth0_user_id": "auth0|664fe2ab91c4fd23c9e12a11",
  "session_type": "guided_practice",
  "target_signs": ["A", "B", "Hello", "I Love You"],
  "attempted_signs": [
    {
      "sign": "A",
      "predicted_sign": "A",
      "confidence_score": 0.97,
      "correct": true
    },
    {
      "sign": "I Love You",
      "predicted_sign": "Hello",
      "confidence_score": 0.61,
      "correct": false
    }
  ],
  "total_attempts": 2,
  "correct_attempts": 1,
  "accuracy": 0.5,
  "duration_seconds": 185,
  "started_at": "2026-04-29T14:15:00Z",
  "ended_at": "2026-04-29T14:18:05Z"
}
```

### Important Fields

- `auth0_user_id`: Connects the practice session to a specific learner.
- `session_type`: Distinguishes guided practice, free practice, or assessment modes.
- `target_signs`: Expected signs for the session.
- `attempted_signs`: Embedded list of each practice attempt and its result.
- `confidence_score`: Shows how strongly the model supported each prediction.
- `correct`: Indicates whether the detected sign matched the target sign.
- `accuracy`: Aggregate session performance metric.
- `duration_seconds`: Useful for engagement and learning analytics.
- `started_at` and `ended_at`: Define the session timeline.

### How It Supports the App

This collection supports the training side of SignLingo. It enables users to practice signs, allows instructors to review improvement over time, and provides measurable outcomes for demonstrating educational value in the capstone project.

---

## 4. `model_metrics`

### Purpose

Stores training, validation, and deployment metadata for machine learning models used by SignLingo.

### Sample JSON Document

```json
{
  "_id": { "$oid": "6639fa0ad0a9e6b8c12a1004" },
  "model_name": "random_forest_sign_classifier",
  "model_version": "v1.0.0",
  "algorithm": "Random Forest",
  "dataset_summary": {
    "total_classes": 32,
    "images_per_class": 850,
    "total_images": 27200,
    "example_labels": ["A", "B", "Hello", "I Love You"]
  },
  "training_metrics": {
    "training_accuracy": 0.991,
    "validation_accuracy": 0.962,
    "precision": 0.958,
    "recall": 0.955,
    "f1_score": 0.956
  },
  "deployment_status": "production",
  "trained_at": "2026-04-20T11:30:00Z",
  "evaluated_at": "2026-04-21T16:45:00Z",
  "notes": "Trained on MediaPipe hand landmarks extracted from the SignLingo image dataset."
}
```

### Important Fields

- `model_name`: Human-readable name of the classifier.
- `model_version`: Version tag used for traceability.
- `algorithm`: Machine learning method used by the model.
- `dataset_summary.total_classes`: Number of supported sign classes.
- `dataset_summary.images_per_class`: Average or target image count per class.
- `training_metrics`: Accuracy and evaluation metrics used in reports.
- `deployment_status`: Indicates whether the model is experimental, staging, or production.
- `trained_at` and `evaluated_at`: Important lifecycle timestamps.
- `notes`: Optional summary of preprocessing or dataset details.

### How It Supports the App

This collection documents the ML side of the project. It supports version control for deployed models, helps compare model performance over time, and provides strong evidence for the technical evaluation section of a capstone presentation.

---

## Summary

The `signlingo_db` database is designed to support four major needs in the SignLingo platform:

- `users` manages identity and profile information.
- `prediction_logs` records real-time translation outputs.
- `practice_sessions` tracks learning activity and user performance.
- `model_metrics` stores machine learning evaluation and deployment metadata.

Together, these collections provide a clean and scalable NoSQL structure for authentication, live inference, analytics, and academic project reporting.