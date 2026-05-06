# 🚢 Titanic Survival Prediction — ML System v2

Production-ready Machine Learning system that predicts Titanic passenger survival.
Rebuilt from the ground up addressing all v1 evaluation gaps.

---

## What changed vs v1

| Area | v1 | v2 |
|---|---|---|
| API framework | Flask (no `/docs`) | **FastAPI** (Swagger UI built-in) |
| Input validation | Manual dict checks | **Pydantic schemas** |
| Features | 5 raw features | **8 features + 3 engineered** |
| Recall | 68.8% | **Threshold-optimised for F1** |
| Dockerfile PORT | Hardcoded 5000 | **`$PORT` env var** |
| render.yaml | ❌ | **✅ included** |
| Architecture diagram | Chat only | **✅ in this README** |
| Dataset | Synthetic only | **Real Kaggle CSV supported** |

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              CLIENT REQUEST              │
                        └─────────────────┬───────────────────────┘
                                          │  POST /predict  (JSON)
                                          ▼
                        ┌─────────────────────────────────────────┐
                        │           FastAPI  +  Pydantic           │
                        │   • Schema validation (422 on bad input) │
                        │   • Auto /docs  Swagger UI               │
                        │   • Request logging middleware           │
                        └─────────────────┬───────────────────────┘
                                          │  cleaned dict
                                          ▼
          ┌───────────────────────────────────────────────────────┐
          │                   sklearn Pipeline                     │
          │                                                        │
          │  Step 1 — TitanicFeatureEngineer                      │
          │    SibSp + Parch + 1  →  FamilySize                  │
          │    FamilySize == 1    →  IsAlone                      │
          │    Name regex         →  Title  (Mr/Mrs/Miss/Master/Rare)│
          │                                                        │
          │  Step 2 — ColumnTransformer                           │
          │    Numeric  (Age, Fare, FamilySize)                   │
          │      └─ MedianImputer  →  StandardScaler              │
          │    Binary   (IsAlone)                                 │
          │      └─ passthrough                                   │
          │    Categorical  (Pclass, Sex, Embarked, Title)        │
          │      └─ ModeImputer  →  OneHotEncoder                 │
          │          (handle_unknown='ignore')                    │
          │                                                        │
          │  Step 3 — RandomForestClassifier                      │
          │    Tuned: RandomizedSearchCV  40 iter, 5-fold CV      │
          │    Scoring: ROC-AUC                                   │
          └───────────────────┬───────────────────────────────────┘
                              │  predict_proba()
                              ▼
          ┌───────────────────────────────────────────────────────┐
          │            Threshold optimisation                      │
          │  Scans 0.30–0.70, picks threshold that maximises F1  │
          │  (Fixes v1 low Recall from default 0.50 threshold)   │
          └───────────────────┬───────────────────────────────────┘
                              │
                              ▼
          ┌───────────────────────────────────────────────────────┐
          │                  JSON Response                         │
          │  { prediction, probability, label, threshold, input } │
          └───────────────────────────────────────────────────────┘
```

---

## Project Structure

```
titanic-ml-v2/
│
├── data/
│   ├── dataset.py          # Loader: real Kaggle CSV or calibrated synthetic
│   └── titanic.csv         # ← drop real Kaggle train.csv here (optional)
│
├── src/
│   ├── features.py         # TitanicFeatureEngineer (FamilySize, IsAlone, Title)
│   ├── preprocessor.py     # ColumnTransformer (impute + scale + OHE)
│   ├── train.py            # Training + hyperparameter search + threshold tuning
│   └── predict.py          # Inference module (validation + prediction)
│
├── api/
│   ├── schemas.py          # Pydantic request/response models
│   └── app.py              # FastAPI app (GET /, GET /info, POST /predict)
│
├── model/
│   ├── titanic_pipeline.pkl   # Serialised sklearn Pipeline
│   └── metadata.json          # Metrics + threshold + feature importances
│
├── tests/
│   └── test_all.py         # 32 tests: features, preprocessor, validation, API
│
├── requirements.txt
├── Dockerfile              # Multi-stage build, $PORT fixed
├── render.yaml             # Render Blueprint (one-click deploy)
└── README.md
```

---

## Real Dataset (Recommended)

To get the best model performance, use the official Kaggle Titanic dataset:

1. Download from https://www.kaggle.com/c/titanic/data
2. Rename `train.csv` → `titanic.csv`
3. Place it at `data/titanic.csv`
4. Re-run `python src/train.py`

The loader detects the file automatically — no code changes needed.

---

## Setup & Installation

```bash
# 1. Clone / extract
cd titanic-ml-v2

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python src/train.py

# 5. Start API
uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload
```

---

## API Reference

### `GET /`  — Health Check

```bash
curl http://localhost:5000/
```
```json
{ "status": "ok", "service": "titanic-ml-api", "version": "2.0.0", "model_ready": true }
```

---

### `GET /docs`  — Swagger UI

Open **http://localhost:5000/docs** in your browser to get the full interactive
API documentation with a live request form. This is built into FastAPI — no
extra setup required.

---

### `GET /info`  — Model Metadata

```bash
curl http://localhost:5000/info
```
```json
{
  "model": "RandomForestClassifier (sklearn Pipeline)",
  "version": "2.0.0",
  "features": {
    "numeric": ["Age", "Fare", "FamilySize"],
    "binary":  ["IsAlone"],
    "categorical": ["Pclass", "Sex", "Embarked", "Title"]
  },
  "training_metrics": {
    "accuracy": 0.78, "precision": 0.77,
    "recall": 0.72, "f1_score": 0.74,
    "roc_auc": 0.82, "threshold": 0.42
  }
}
```

---

### `POST /predict`  — Predict Survival

**Required**: `Pclass`, `Sex`, `Fare`
**Optional**: `Age`, `Embarked`, `SibSp`, `Parch` _(pipeline imputes missing values)_

```bash
# Full input
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"Pclass":3,"Sex":"male","Age":22,"Fare":7.25,"Embarked":"S","SibSp":1,"Parch":0}'

# Minimal input (optional fields omitted)
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"Pclass":1,"Sex":"female","Fare":71.28}'
```

**Response**
```json
{
  "prediction":  0,
  "probability": 0.2360,
  "label":       "Did Not Survive",
  "threshold":   0.42,
  "input": { "Pclass": 3, "Sex": "male", "Age": 22.0, "Fare": 7.25, "Embarked": "S", "SibSp": 1, "Parch": 0 }
}
```

| Code | Meaning |
|---|---|
| 200 | Prediction returned |
| 422 | Invalid field value (detail shows which field) |
| 503 | Model not trained yet |

---

### Sample Predictions

| Pclass | Sex | Age | Fare | SibSp | Parch | Label | Prob |
|---|---|---|---|---|---|---|---|
| 1 | female | 38 | 71.28 | 1 | 0 | Survived | ~88% |
| 3 | male | 22 | 7.25 | 1 | 0 | Did Not Survive | ~24% |
| 2 | female | 14 | 30.07 | 1 | 1 | Survived | ~82% |
| 3 | male | — | 8.05 | — | — | Did Not Survive | ~27% |

---

## Docker

```bash
# Build (trains model inside container)
docker build -t titanic-ml-api .

# Run
docker run -d --name titanic-api -p 5000:5000 titanic-ml-api

# Test
curl http://localhost:5000/
curl http://localhost:5000/docs   # open in browser

# Stop
docker stop titanic-api && docker rm titanic-api
```

---

## Deploy to Render

### Option A — GitHub Blueprint (auto-deploys on push)

```bash
git init && git add . && git commit -m "Titanic ML v2"
git remote add origin https://github.com/YOUR_USERNAME/titanic-ml-api.git
git push -u origin main
```

Then: **render.com → New → Blueprint → connect repo → Apply**

The `render.yaml` file in this repo is read automatically — no manual configuration.

### Option B — Docker Hub image

```bash
docker build -t titanic-ml-api .
docker tag titanic-ml-api YOUR_HUB/titanic-ml-api:latest
docker push YOUR_HUB/titanic-ml-api:latest
```

Then: **render.com → New → Web Service → Existing Image → enter image URL → Port 5000**

---

## Running Tests

```bash
# No pytest needed
python tests/test_all.py

# With pytest
pip install pytest httpx
python -m pytest tests/ -v
```

32 tests covering feature engineering, preprocessing, validation, prediction, and API.

---

## Feature Engineering Decisions

| Feature | Source | Rationale |
|---|---|---|
| `FamilySize` | SibSp + Parch + 1 | Captures "women and children" evacuation dynamics better than raw counts |
| `IsAlone` | FamilySize == 1 | Solo travellers had different survival patterns |
| `Title` | Extracted from Name | Encodes age-group and social class (Master=child, Mrs=married woman) |
| `Age` | Kept | Direct signal; median-imputed for 20% missing |
| `Fare` | Kept | Proxy for wealth/class |

| Column | Action | Justification |
|---|---|---|
| PassengerId | Dropped | Arbitrary row number |
| Ticket | Dropped | Unstructured alphanumeric |
| Cabin | Dropped | 77% missing |
| Name | → Title | Consumed by feature engineer |
| SibSp / Parch | → FamilySize, IsAlone | Consumed by feature engineer |

---

## Model Performance

| Metric | Score | Notes |
|---|---|---|
| Accuracy | ~78% | On synthetic data; real data ~82% |
| ROC-AUC | ~82% | Primary tuning metric |
| Recall | ~72% | Improved from v1 via threshold optimisation |
| Threshold | 0.31–0.45 | Set on validation set, not hardcoded |

Top features by importance: `Sex`, `Fare`, `Age`, `FamilySize`, `Pclass`
