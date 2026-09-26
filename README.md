# DeepFake-Detection .

This Repository i used for upload deep fake detection model .

# Deepfake Detection Model

A machine learning model that classifies media (image, video, or audio) as **Real** or **Fake** using metadata and signal-quality features rather than raw pixel/audio analysis.

## Overview

This project trains a **Logistic Regression** classifier on a metadata dataset describing media samples (e.g. lip-sync consistency, visual artifacts, lighting inconsistencies, compression level, platform, and generation method) to predict whether the sample is real or AI-generated ("fake").

## Dataset

The model expects a CSV file named `deepfake_detection_metadata_dataset.csv` with the following columns:

| Column | Description |
|---|---|
| `media_id` | Unique identifier for the media sample |
| `media_type` | Type of media (Image, Video, Audio) |
| `content_category` | Content category (News, Interview, Social Media, etc.) |
| `face_count` | Number of faces detected in the media |
| `audio_present` | Whether audio is present (Yes/No) |
| `lip_sync_score` | Score indicating lip-sync consistency |
| `visual_artifacts_score` | Score indicating presence of visual artifacts |
| `compression_level` | Compression level of the media file |
| `lighting_inconsistency_score` | Score indicating lighting inconsistencies |
| `source_platform` | Platform the media was sourced from |
| `generation_method` | Method used to generate fake media (GAN, Diffusion, VoiceClone, etc.), if applicable |
| `label` | Target variable — `Real` or `Fake` |

## Pipeline

1. **Load & explore data** — inspect shape, head/tail, dtypes, and missing values.
2. **Handle missing values** — `generation_method` (missing for real media) is imputed with its mode.
3. **Encode categorical features** — one-hot encoding via `pd.get_dummies()` on `media_type`, `content_category`, `audio_present`, `source_platform`, and `generation_method`.
4. **Train/test split** — 80/20 split (`random_state=42`).
5. **Train model** — `sklearn.linear_model.LogisticRegression`.
6. **Evaluate** — confusion matrix and accuracy score on the held-out test set.
7. **Visualize** — box plots comparing `lip_sync_score`, `visual_artifacts_score`, and `lighting_inconsistency_score` between real and fake samples.
8. **Save model** — trained model is serialized to `deepfake_model.pkl` with `joblib`.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Usage

1. Place `deepfake_detection_metadata_dataset.csv` in the project directory.
2. Run the notebook (`deepfake_model.ipynb`) cell by cell, or export it to a script:
   ```bash
   jupyter nbconvert --to script deepfake_model.ipynb
   python deepfake_model.py
   ```
3. The trained model is saved as `deepfake_model.pkl`, and box plot figures are saved as `<feature>_boxplot.png`.

### Loading the saved model

```python
import joblib

model = joblib.load("deepfake_model.pkl")
predictions = model.predict(X_new)
```

## Results

The model achieves high accuracy on the test split (confusion matrix and accuracy score are printed in the notebook).

> **Note:** Accuracy came out at 100% on this dataset, which is unusually high for a real-world classifier. This is very likely because `generation_method` is only missing for `Real` samples (and filled in for `Fake` samples), so its one-hot encoding leaks the label. Before using this model on new data, consider dropping or re-deriving `generation_method` so the model relies on genuine signal-quality features (`lip_sync_score`, `visual_artifacts_score`, `lighting_inconsistency_score`, `compression_level`) instead.

## Limitations & Future Work

- Trained on metadata/scores rather than raw media (no image, video frame, or audio waveform analysis).
- Potential label leakage via `generation_method` (see note above) should be addressed before deployment.
- No hyperparameter tuning, cross-validation, or regularization scaling was applied (`LogisticRegression` also raised a convergence warning — consider scaling features or increasing `max_iter`).
- Could be extended with a CNN/RNN pipeline on raw video frames or audio spectrograms for true content-based deepfake detection.

## License

Add a license of your choice (e.g. MIT) here.
