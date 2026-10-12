# Thermal Vessel Detection, Classification and Re-ID

A pipeline that uses a **thermal observation camera** to:

1. **Detect** vessels in video
2. **Classify** each vessel's type from any viewing angle
3. **Re-identify** vessels: decide whether a detection is the same boat as one already saved, and link it to that vessel's stored information with a calibrated confidence score

> **Status:** early development. The structure, commands, and thresholds below are the plan and will be updated as the code and data land. See the project planning doc for the full rationale.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Data](#data)
- [Training](#training)
- [Running the Pipeline](#running-the-pipeline)
- [Vessel Database and Matching](#vessel-database-and-matching)
- [API](#api)
- [Evaluation](#evaluation)
- [Roadmap](#roadmap)
- [Known Limitations](#known-limitations)
- [Team](#team)
- [References](#references)

---

## How It Works

```
Thermal video
   -> Detector (YOLO)            finds vessels in each frame
   -> Tracker (ByteTrack)        assigns a track ID within a video
   -> Crop selection             keeps the best 5-10 crops per track
   -> Classifier                 predicts vessel type
   -> Re-ID model                produces an embedding per crop (averaged per track)
   -> Gallery search             nearest neighbors in the vessel database
   -> Decision logic             match / possible match / new vessel
   -> API + UI                   human confirms or corrects; confirmed data is enrolled
```

The three learned models (detector, classifier, re-ID) are **trained separately** and chained only at inference. The tracker needs tuning but no training.

### Matching decisions

Raw cosine similarity is not a probability, so scores are **calibrated** (Platt scaling or isotonic regression) on validation pairs before being shown as a percentage.

| Calibrated confidence | Decision | Behavior |
|---|---|---|
| >= 90% (tunable) | Match | Auto-link and show vessel info |
| 60-90% | Possible match | Show candidates for human confirmation |
| < 60% | New vessel | Create an unverified record |

A margin check (top-1 must beat top-2 by a set amount) and a type/size cross-check reduce wrong matches.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Language / framework | Python 3.11, PyTorch |
| Detection | Ultralytics YOLO (v8/11), OBB variant if needed |
| Classification | timm (ConvNeXt / ViT) or DINOv2 features |
| Tracking | ByteTrack / BoT-SORT |
| Re-ID | torchreid or FastReID, pytorch-metric-learning |
| Embedding search | pgvector (PostgreSQL) or FAISS |
| Video I/O | OpenCV, FFmpeg / GStreamer |
| Labeling | CVAT |
| Augmentation | Albumentations |
| Experiment tracking | Weights & Biases or TensorBoard |
| Backend | FastAPI |
| Database | PostgreSQL |
| Frontend | Streamlit (prototype), React (later) |
| Packaging | Docker, uv or conda |

---

## Repository Structure

Planned layout:

```
.
├── configs/                 # YAML configs for each model and the pipeline
├── data/
│   ├── raw/                 # original thermal videos (not committed)
│   ├── detection/           # images + labels for the detector
│   ├── classification/      # crops sorted by vessel type
│   └── reid/                # crops sorted by vessel identity
├── src/
│   ├── detection/           # detector training and inference
│   ├── tracking/            # tracker wrapper and tuning
│   ├── classification/      # classifier training and inference
│   ├── reid/                # re-ID training, embedding, calibration
│   ├── matching/            # gallery search and decision logic
│   ├── pipeline/            # end-to-end inference
│   └── api/                 # FastAPI app
├── app/                     # UI (Streamlit / React)
├── db/                      # SQL schema and migrations
├── scripts/                 # data prep, splitting, evaluation helpers
├── notebooks/               # exploration and error analysis
├── tests/
├── docker/
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11
- A CUDA-capable GPU (8-16 GB is enough for training; Colab or Kaggle also work)
- PostgreSQL 15+ with the `pgvector` extension (for the vessel database)
- FFmpeg

### Installation

```bash
git clone <repo-url>
cd <repo-name>

# using conda
conda create -n vessel python=3.11 -y
conda activate vessel
pip install -r requirements.txt
```

Or with Docker:

```bash
docker compose -f docker/docker-compose.yml up --build
```

### Database setup

```bash
createdb vessels
psql vessels -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql vessels -f db/schema.sql
```

---

## Data

### Thermal capture guidelines

Consistency matters more than volume. Before recording:

- Pick **one palette** (grayscale white-hot or black-hot) and use it for **all** data. Never mix palettes.
- Lock gain/AGC settings where the device allows, and log them.
- Record the camera's mounting height, angle, and field of view (needed for size estimation).
- Check whether the device outputs radiometric (temperature) data or only processed video, and record in the richest format available.
- Cover day and night, seasons, engines on and off, near and far, and all viewing angles.

### Labeling

- **Detection:** boxes in CVAT, one class (`vessel`) to start.
- **Classification:** vessel type per crop; start with coarse classes (small boat, fishing, cargo, tanker, military, sailboat).
- **Re-ID:** identity ID per vessel. Tracker output gives within-video identities; merge tracks of the same boat across videos manually.

Follow the labeling guideline in the planning doc and review samples as a team.

### Splitting (important)

**Split by video or by boat identity, never by random frame.** Neighboring frames are nearly identical, so random splits leak test data and produce fake accuracy. For re-ID, test identities must be completely unseen during training.

Suggested split: 70% train / 15% validation / 15% test.

### Preprocessing

- Convert single-channel thermal to three channels by replication (or modify the first conv layer to accept one channel).
- Normalize consistently (e.g., clip to the 1st-99th percentile, then scale).
- Use the same preprocessing at training and inference.

### Public datasets worth evaluating

VAIS (visible + infrared ships), FLIR ADAS (generic thermal pretraining), Singapore Maritime Dataset, SeaShips, MARVEL. Most are visible-light, so expect a domain gap; your own thermal footage is the main data source.

---

## Training

Train in this order. Exact commands will be finalized as the scripts are written.

### 1. Detector

```bash
python src/detection/train.py --config configs/detector.yaml
```

- Start from pretrained COCO/DOTA weights.
- Use a large image size (960-1280); boats are often small.
- Reduce hue augmentation (no color); keep scale, rotation, brightness/contrast, blur, and noise.
- Include 5-10% background images (waves, buoys, glints, birds) to reduce false positives.
- Metrics: mAP@0.5, mAP@0.5:0.95, precision, recall.

### 2. Vessel type classifier

```bash
python src/classification/train.py --config configs/classifier.yaml
```

- Pretrained timm backbone with a new head.
- Phase 1: frozen backbone, train the head (~5 epochs, lr ~1e-3).
- Phase 2: unfreeze all (lr 1e-5 to 5e-5, cosine decay, warmup, 20-50 epochs).
- Cross-entropy with label smoothing 0.1; class weights if imbalanced.
- Metrics: top-1, macro F1, confusion matrix.

### 3. Re-ID model

```bash
python src/reid/train.py --config configs/reid.yaml
```

- Backbone (ResNet50-IBN / ViT / TransReID) + BNNeck + 512/768-dim embedding.
- Loss: identity classification (label smoothing) + batch-hard triplet loss, or ArcFace.
- PK sampler (e.g., 16 identities x 4 images per batch).
- Include the same boats at different times of day so the model learns identity, not time-of-day heat patterns.
- Mine hard negatives (different boats that look alike) after the first run.
- Metrics: Rank-1, mAP, ROC for same/different pairs.

### 4. Calibration

```bash
python src/reid/calibrate.py --config configs/reid.yaml
```

Fits similarity-to-probability mapping on validation pairs and saves the thresholds used by the decision logic.

### 5. Tracker

No training. Tune confidence threshold, track buffer, and IoU threshold on validation videos and check ID switches (IDF1 / MOTA via TrackEval).

---

## Running the Pipeline

On a recorded video:

```bash
python src/pipeline/run.py \
  --video path/to/thermal.mp4 \
  --detector weights/detector.pt \
  --classifier weights/classifier.pt \
  --reid weights/reid.pt \
  --out outputs/
```

On a live stream:

```bash
python src/pipeline/run.py --source rtsp://<camera-address>/stream ...
```

Start the API and UI:

```bash
uvicorn src.api.main:app --reload
streamlit run app/main.py
```

---

## Vessel Database and Matching

### Tables

- **vessels**: `vessel_id`, `name`, `vessel_type`, `length_m`, `notes`, `first_seen`, `last_seen`, `status` (confirmed / unverified)
- **vessel_embeddings**: `embedding_id`, `vessel_id`, `embedding`, `image_path`, `captured_at`, `camera_id`, `view_angle`, `quality`, `model_version`
- **sightings**: `sighting_id`, `track_id`, `matched_vessel_id`, `similarity`, `decision`, `captured_at`

One vessel has many embeddings (different angles, times, conditions).

### Rules for keeping the database trustworthy

- Enroll embeddings **only** from human-confirmed or very-high-confidence matches. Never auto-add low-confidence ones.
- Store several embeddings per vessel covering different angles and times of day.
- Filter out blurry, tiny, or heavily occluded crops.
- When the re-ID model is retrained, **re-embed the whole database** (keep raw crops and the `model_version` column).

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /match` | Takes a track or crops, returns ranked candidate vessels |
| `POST /confirm` | Links a sighting to a vessel and enrolls its embeddings |
| `POST /vessels` | Creates a new vessel record |
| `GET /vessels/{id}` | Returns vessel info and past sightings |

Example `/match` response:

```json
{
  "track_id": "t_0412",
  "decision": "possible",
  "candidates": [
    {
      "vessel_id": "...",
      "name": "Example Vessel",
      "type": "fishing",
      "confidence": 0.84,
      "last_seen": "2026-10-09T22:14Z"
    }
  ]
}
```

---

## Evaluation

Per-model metrics are listed under [Training](#training). Because errors compound across stages, also run an **end-to-end benchmark**:

1. Hold out some boats entirely as "unknown"; enroll the rest as "known."
2. Run detector -> tracker -> classifier + re-ID on held-out test videos using **detected** crops (not ground-truth crops).
3. Measure top-1 and top-5 accuracy on known boats and the **false-accept rate on unknown boats** at the chosen threshold.
4. Plot the trade-off curve and set thresholds from it.

End-to-end numbers will be lower than isolated stage numbers. If so, fine-tune the classifier and re-ID model on detector-produced crops.

---

## Roadmap

| Phase | Weeks | Deliverables |
|---|---|---|
| 1. Setup and data | 1-2 | Environment, repo, labeling guideline, first footage, splits, baseline YOLO |
| 2. Detection | 2-3 | Thermal-tuned detector, tracker running, crops exported |
| 3. Classification | 3-4 | Coarse type classifier, first end-to-end detect-track-classify demo |
| 4. Re-ID | 4-7 | Identity labels, trained re-ID model, calibration, hard-negative round |
| 5. Database and app | 6-8 | Postgres schema, matching API, confirmation UI |
| 6. Integration and evaluation | 8-10 | End-to-end benchmark, threshold tuning, error analysis |
| 7. Stretch | 10+ | Real-time / edge deployment (Jetson, TensorRT), visible-camera fusion |

---

## Known Limitations

- Thermal imagery has **no color and no readable hull markings**; re-ID relies on shape, size, superstructure layout, and heat signature.
- The same boat can look different at different times of day (engine state, sun heating).
- Near-identical boats of the same class are the hardest re-ID case. Matches are probabilistic and need human confirmation for anything below the auto-match threshold.
- Public thermal maritime data is scarce; performance depends heavily on our own footage.
- Real-time operation is a stretch goal; offline processing of recorded video is the baseline.

---

## Team

| Role | Responsibilities |
|---|---|
| Detection and tracking | Detector training, tracker tuning, false-positive analysis |
| Classification and datasets | Labeling guidelines, crop datasets, classifier training |
| Re-ID and embedding search | Identity data, re-ID training, calibration, gallery search |
| Backend and UI | Database, FastAPI, interface, confirmation workflow |

Add names here.

---

## References

**Detection and tracking:** Ultralytics YOLO docs; RT-DETR / DETR; Oriented R-CNN, ReDet, MMRotate; ByteTrack, BoT-SORT, DeepSORT

**Classification:** fine-grained ship classification surveys; Vision Transformers, DINOv2, CLIP

**Re-identification:** *Bag of Tricks and a Strong Baseline for Deep Person Re-identification*; *In Defense of the Triplet Loss for Person Re-Identification*; TransReID; CLIP-ReID; vehicle re-ID surveys and VeRi-776; visible-infrared re-ID papers; torchreid, FastReID, pytorch-metric-learning

**Thermal and maritime:** thermal infrared object detection and infrared ship detection surveys; RGB-to-infrared domain adaptation; VAIS, FLIR ADAS, Singapore Maritime Dataset, SeaShips, MARVEL

---

## License

Add a license (e.g., MIT) before sharing the repository.
