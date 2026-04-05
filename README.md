# 🛰️ SPACE WEATHER → SATELLITE RISK PREDICTION

## 📁 Project Structure

```
space_weather_project/
│
├── data/
│   ├── nasa_omni_space_weather.csv     ← REAL dataset (included!)
│   └── preprocess_data.py             ← STEP 1: Clean + engineer features
│
├── models/
│   └── train_models.py                ← STEP 2: Train all ML models + PCA
│
├── dashboard/
│   └── app.py                         ← STEP 3: Launch dashboard
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Project (3 Steps)

### Step 1 — Preprocess the dataset
```bash
python data/preprocess_data.py
```
Reads nasa_omni_space_weather.csv, cleans it, creates risk labels,
engineers features, saves processed CSV files.

### Step 2 — Train ML models
```bash
python models/train_models.py
```
Trains Random Forest, XGBoost, Logistic Regression (PCA), Isolation Forest.
Saves all .pkl model files.

### Step 3 — Launch dashboard
```bash
streamlit run dashboard/app.py
```
Opens at http://localhost:8501

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | nasa_omni_space_weather.csv |
| Rows | 10,000 hourly measurements |
| Features | 14 raw columns |
| After engineering | 28 total features |
| Source | NASA OMNI-style space weather parameters |

### Columns in dataset
- solar_wind_speed, proton_density, bx, by, bz
- plasma_temp, kp_index, dst_index, xray_flux, ae_index
- f107_index (solar radio flux), ap_index (geomagnetic Ap)
- speed_pressure (dynamic pressure), timestamp

---

## 🏷️ Risk Labels

| Condition | Risk |
|-----------|------|
| Kp > 5 or Dst < -100 nT | 🔴 HIGH |
| Kp 3-5 or Dst -50 to -100 | 🟡 MEDIUM |
| Kp < 3 and Dst > -50 | 🟢 LOW |

---

## 🧠 ML Models

| Model | Features Used | Purpose |
|-------|--------------|---------|
| Random Forest | All 28 features | Primary classifier |
| XGBoost | All 28 features | Highest accuracy |
| Logistic Regression | PCA components | Baseline + PCA demo |
| Isolation Forest | All 28 features | Anomaly detection |

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| ModuleNotFoundError | pip install -r requirements.txt |
| Dataset not found | Make sure nasa_omni_space_weather.csv is in data/ folder |
| Models not found | Run Steps 1 and 2 first |
| Empty dataset error | Run python data/preprocess_data.py again |
