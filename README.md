# Thermal Vessel Detection, Classification and Re-ID

A pipeline that uses a **thermal observation camera** to:

1. **Detect** vessels in thermal images
2. **Classify** each vessel's type from any viewing angle
3. **Re-identify** vessels: decide whether a detection is the same boat as one already saved, and link it to that vessel's stored information with a calibrated confidence score

> **Status:** image-first setup. CPU dependencies, dataset folders, validation, and
> a YOLO smoke-training configuration are available. Custom classification, re-ID,
> calibrated matching, API, UI, and video tracking remain planned.

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
Thermal image
   -> Detector (YOLO)            finds vessels in each image
   -> Vessel crops               one crop per detected vessel
   -> Classifier                 predicts vessel type
   -> Re-ID model                produces an embedding per crop
   -> Gallery search             nearest neighbors in the vessel database
   -> Decision logic             match / possible match / new vessel
   -> API + UI                   human confirms or corrects; confirmed data is enrolled
```

The three learned models (detector, classifier, re-ID) are **trained separately**
and will be chained at inference. Images do not need a tracker; ByteTrack and
video processing are deferred. Only the detector workflow is available now.

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
| Language / framework | Python 3.11+ (local setup tested on 3.13), CPU PyTorch + torchvision |
| Detection | Ultralytics YOLO (v8/11), OBB variant if needed |
| Classification | timm (model implementation planned) |
| Tracking | ByteTrack / BoT-SORT (future video phase) |
| Re-ID | pytorch-metric-learning (model implementation planned) |
| Embedding search | FAISS CPU; pgvector deferred |
| Image I/O | Pillow, OpenCV |
| Labeling | CVAT |
| Calibration / evaluation | scikit-learn, NumPy, Matplotlib |
| Testing | pytest |
| Backend / database / UI | FastAPI, PostgreSQL, Streamlit (deferred) |
| Environment | Local venv; CUDA / Docker deferred |

---

## Repository Structure

Initial scaffold:

```
.
├── README.md
├── configs/                 # YOLO data and CPU smoke-training YAML
├── data/
│   ├── raw/                 # original thermal images (not committed)
│   ├── detection/           # images + labels for the detector
│   ├── classification/      # crops sorted by vessel type
│   └── reid/                # crops sorted by vessel identity
├── src/
│   ├── detection/            # detector training and inference
│   ├── tracking/             # tracker wrapper and tuning
│   ├── classification/       # classifier training and inference
│   ├── reid/                 # re-ID training, embedding, calibration
│   ├── matching/             # gallery search and decision logic
│   ├── pipeline/             # end-to-end inference
│   └── api/                  # FastAPI app
├── app/                      # UI (Streamlit / React)
├── db/                       # SQL schema and migrations
├── scripts/                  # data prep, splitting, evaluation helpers
├── notebooks/                # exploration and error analysis
├── tests/
├── requirements.txt          # image-first ML dependencies
├── pyproject.toml            # pytest configuration
└── docker/
```

Data, weights, virtual environments, and generated outputs are ignored by Git.
Empty data directories are retained with `.gitkeep` files. The custom model
packages and deployment directories are placeholders, not implemented services.

---

## Getting Started

### Prerequisites

- Python 3.11 or newer; the existing local environment uses Python 3.13.
- CPU is enough for setup and a tiny training smoke test. Use a GPU later for
  meaningful training; no CUDA toolkit is required for this setup.
- PostgreSQL, FFmpeg, Docker, and a camera are **not required** for the image phase.

### Installation

Run these PowerShell commands from the repository root. The local `.venv`
already exists; on a fresh checkout, create it first:

```powershell
py -3.13 -m venv .venv
```

Install the matching CPU PyTorch pair first, then the remaining dependencies.
Using the interpreter explicitly avoids PowerShell activation-policy issues:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" "torchvision>=0.21,<1" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scripts.check_environment
.\.venv\Scripts\python.exe -m pytest
```

Select `.venv\Scripts\python.exe` using **Python: Select Interpreter** in VS Code.
The environment check imports every direct dependency and exercises torchvision
CPU NMS; it does not download weights or require a dataset.

Version ranges in `requirements.txt` are compatibility bounds, not an exact lock
file. Re-run the checks after dependency updates. When moving to an NVIDIA GPU,
replace torch and torchvision together using the appropriate command from the
official [PyTorch installer](https://pytorch.org/get-started/locally/); do not mix
CPU and CUDA builds. Database and Docker setup will be added with those features.

---

## Data

### Image dataset layout

Put original images in `data/raw/`. Export detection annotations from CVAT in
**YOLO detection** format and arrange them as follows:

```text
data/
  detection/
    images/
      train/                 # image_001.png, ...
      val/
      test/
    labels/
      train/                 # image_001.txt, ...
      val/
      test/
  classification/
    train/<vessel_type>/     # cropped images, e.g. fishing, cargo
    val/<vessel_type>/
    test/<vessel_type>/
  reid/
    train/<vessel_id>/       # multiple crops of the same identified vessel
    val/<vessel_id>/
    test/<vessel_id>/
```

The split directories exist; create class and identity folders when those labels
are known. Detection uses whole images; classification and re-ID use individual
vessel crops. Re-ID needs confirmed identity labels and multiple views per vessel,
not just type labels. Keep unknown identities out of re-ID training until labeled.

Each detection image needs a same-stem `.txt` file at the matching relative path.
Each annotation line is `0 x_center y_center width height`, with coordinates
normalized to image dimensions. Class `0` means `vessel`. Use an **empty label
file** for a verified background-only image. Supported validator formats are
PNG, JPEG, BMP, and TIFF; start with consistently processed 8-bit PNG/JPEG.
Preserve radiometric originals separately; 16-bit inputs require a deliberate
conversion policy before using the baseline training workflow.

```powershell
.\.venv\Scripts\python.exe -m scripts.validate_detection_data
```

Validation fails on empty splits, missing/orphan labels, unreadable images,
invalid boxes, and byte-identical images across splits. It does not detect
near-duplicates or prove identity-level separation; review those manually.
No images are bundled, so validation will fail until all three splits are populated.

### Thermal capture guidelines

Consistency matters more than volume. When collecting images:

- Pick **one palette** (grayscale white-hot or black-hot) and use it for **all** data. Never mix palettes.
- Lock gain/AGC settings where the device allows, and log them.
- Record the camera's mounting height, angle, and field of view (needed for size estimation).
- Check whether the device outputs radiometric (temperature) data or processed images, and preserve the richest format available.
- Cover day and night, seasons, engines on and off, near and far, and all viewing angles.

### Labeling

- **Detection:** boxes in CVAT, one class (`vessel`) to start.
- **Classification:** vessel type per crop; start with coarse classes (small boat, fishing, cargo, tanker, military, sailboat).
- **Re-ID:** manually confirmed identity ID per vessel, consistent across images and capture sessions.

Review sample annotations as a team before labeling a large dataset.

### Splitting (important)

**Split by capture session or boat identity, not by random image.** Burst images
and near-duplicates must stay together. For re-ID, validation/test identities must
be unseen during training; create query/gallery subsets within evaluation
identities later. Do not place the same identity across train, validation, and test.

Suggested split: 70% train / 15% validation / 15% test.

### Preprocessing

- Convert single-channel thermal to three channels by replication (or modify the first conv layer to accept one channel).
- Normalize consistently (e.g., clip to the 1st-99th percentile, then scale).
- Use the same preprocessing at training and inference.

### Public datasets worth evaluating

VAIS (visible + infrared ships), FLIR ADAS (generic thermal pretraining), Singapore Maritime Dataset, SeaShips, MARVEL. Most are visible-light, so expect a domain gap; your own thermal images are the main data source.

---

## Training

Start with detection. Classification, re-ID, and calibration below describe
future work, not runnable training scripts.

### 1. Detector

After populating and validating the dataset, run the CPU smoke test:

```powershell
.\.venv\Scripts\yolo.exe detect train cfg=configs/detector-smoke.yaml
```

The first run downloads pretrained `yolo11n.pt` weights. One epoch at 320 pixels
checks the workflow only; it is not a trained thermal-vessel model. Results are
saved under `outputs/detection/smoke` (Ultralytics may increment the directory
name on subsequent runs). Zero workers avoids multiprocessing overhead on Windows.

- Start from pretrained COCO weights.
- Later use a larger image size (960-1280); boats are often small.
- Reduce hue augmentation (no color); keep scale, rotation, brightness/contrast, blur, and noise.
- Include 5-10% background images (waves, buoys, glints, birds) to reduce false positives.
- Metrics: mAP@0.5, mAP@0.5:0.95, precision, recall.

### 2. Vessel type classifier

- Pretrained timm backbone with a new head.
- Phase 1: frozen backbone, train the head (~5 epochs, lr ~1e-3).
- Phase 2: unfreeze all (lr 1e-5 to 5e-5, cosine decay, warmup, 20-50 epochs).
- Cross-entropy with label smoothing 0.1; class weights if imbalanced.
- Metrics: top-1, macro F1, confusion matrix.

### 3. Re-ID model

- Backbone (ResNet50-IBN / ViT / TransReID) + BNNeck + 512/768-dim embedding.
- Loss: identity classification (label smoothing) + batch-hard triplet loss, or ArcFace.
- PK sampler (e.g., 16 identities x 4 images per batch).
- Include the same boats at different times of day so the model learns identity, not time-of-day heat patterns.
- Mine hard negatives (different boats that look alike) after the first run.
- Metrics: Rank-1, mAP, ROC for same/different pairs.

### 4. Calibration

Will fit similarity-to-probability mapping on validation pairs and save the
thresholds used by the decision logic. FAISS similarity alone is not confidence.

### 5. Tracker

Deferred until video support. No tracker is used with the current image datasets.

---

## Running the Pipeline

### Current image detection baseline

After the smoke training run, use its actual `best.pt` path:

```powershell
.\.venv\Scripts\yolo.exe detect predict model=outputs/detection/smoke/weights/best.pt source=data/detection/images/test device=cpu save=True save_crop=True project=outputs/detection name=predict
```

This saves annotated images and vessel crops. Generic COCO pretrained weights
are not a custom single-class thermal-vessel detector; inspect baseline results
and train on your annotations before relying on them.

The end-to-end classification/re-ID pipeline, API, UI, and live/video sources are
not implemented yet.

---

## Vessel Database and Matching

### Tables

Planned schema; no database is needed for the current image setup.

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

Planned endpoints; no API server is implemented yet.

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
2. Run detector -> classifier + re-ID on held-out test images using **detected** crops (not ground-truth crops).
3. Measure top-1 and top-5 accuracy on known boats and the **false-accept rate on unknown boats** at the chosen threshold.
4. Plot the trade-off curve and set thresholds from it.

End-to-end numbers will be lower than isolated stage numbers. If so, fine-tune the classifier and re-ID model on detector-produced crops.

---

## Roadmap

| Phase | Weeks | Deliverables |
|---|---|---|
| 1. Setup and data | 1-2 | CPU environment, image labels, session/identity-safe splits, baseline YOLO |
| 2. Detection | 2-3 | Thermal-tuned image detector, crops exported |
| 3. Classification | 3-4 | Coarse type classifier, detect-classify image demo |
| 4. Re-ID | 4-7 | Identity labels, trained re-ID model, calibration, hard-negative round |
| 5. Database and app | 6-8 | Postgres schema, matching API, confirmation UI |
| 6. Integration and evaluation | 8-10 | End-to-end benchmark, threshold tuning, error analysis |
| 7. Stretch | 10+ | Video ingestion/tracking, real-time / edge deployment, visible-camera fusion |

---

## Known Limitations

- Thermal imagery has **no color and no readable hull markings**; re-ID relies on shape, size, superstructure layout, and heat signature.
- The same boat can look different at different times of day (engine state, sun heating).
- Near-identical boats of the same class are the hardest re-ID case. Matches are probabilistic and need human confirmation for anything below the auto-match threshold.
- Public thermal maritime data is scarce; performance depends heavily on our own images.
- CPU training is for smoke tests, not a substitute for a GPU for full experiments.
- Real-time and video processing are deferred; still images are the current baseline.

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
