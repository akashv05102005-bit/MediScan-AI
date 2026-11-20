# MediScan AI
AI-assisted chest X-ray / CT analysis with **transfer learning** (ResNet50, EfficientNet-B0, DenseNet121) and **Grad-CAM** explainability, delivered through a professional clinical dashboard.

> Research and educational prototype. Not a certified medical device. All output must be reviewed by a qualified clinician.

## Dashboard features
- Patient and study details (ID, age, sex, modality, clinical history)
- Upload JPG / PNG / BMP / **DICOM (.dcm)**
- AI finding with confidence ring and **triage suggestion** (Routine / Expedited / Priority / Manual review)
- **Grad-CAM heatmap** for the predicted class or any other class
- **Image-quality checks** (resolution, brightness, contrast, sharpness, radiograph-like)
- **Clinician review** (agree / disagree / inconclusive + comments)
- **PDF report** download, heatmap PNG download
- Session log with CSV export
- Model card and real measured performance (confusion matrix, ROC, sensitivity / specificity)

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
Recommended: *Chest X-Ray Images (Pneumonia)* or the *COVID-19 Radiography Database* from Kaggle.
```
data/
  train/NORMAL/*.jpeg   train/PNEUMONIA/*.jpeg   (more classes allowed)
  test/NORMAL/*.jpeg    test/PNEUMONIA/*.jpeg    (optional, recommended)
```
A stratified 15% validation split is created automatically from `train`.
To smoke-test the pipeline only: `python scripts/make_dummy_data.py`

## Train
```bash
python -m src.train --model efficientnet_b0 --batch-size 16 --epochs-head 2 --epochs-ft 3 --workers 0
```
Outputs in `outputs/`: `best_model.pt`, `training_curves.png`, `confusion_test.png`, `roc_test.png`, `metrics_test.json`.

## Run
```bash
streamlit run app.py
python -m src.evaluate
python -m src.predict --image sample.jpeg --out result.png
```

## Project structure
```
app.py                Streamlit dashboard
.streamlit/config.toml  theme and server settings
src/config.py         paths, constants, disclaimer
src/dataset.py        loaders, augmentation, stratified split
src/model.py          backbones, freezing, Grad-CAM target layers
src/train.py          two-phase training loop
src/evaluate.py       metrics and plots
src/gradcam.py        Grad-CAM and overlay
src/predict.py        single-image inference
src/quality.py        image-quality checks
src/report.py         PDF report generation
```

## Limitations
Not externally validated. Performance depends entirely on the training data. Heatmaps explain the model, not the patient.
