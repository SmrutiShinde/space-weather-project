"""
STEP 2: train_models.py
========================
Loads the processed real NASA OMNI space weather dataset and trains:
<<<<<<< HEAD
  - Random Forest (on PCA-reduced features)
  - XGBoost (on PCA-reduced features)
  - Logistic Regression (on PCA-reduced features)
  - Isolation Forest (anomaly detection)

KEY IMPROVEMENT: Uses Ordinal Encoding instead of LabelEncoder.
  LabelEncoder encodes alphabetically: High=0, Low=1, Medium=2
  This BREAKS the natural order of risk levels.

  Ordinal Encoding preserves natural order:
  Low=0  <  Medium=1  <  High=2  ✅ Correct!

=======
  - Random Forest
  - XGBoost
  - Logistic Regression (on PCA-reduced features)
  - Isolation Forest (anomaly detection)

>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
Also applies PCA for dimensionality reduction and visualization.
Saves all models + artifacts to /models directory.

Run AFTER preprocess_data.py
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from sklearn.preprocessing import MinMaxScaler
=======
from sklearn.preprocessing import LabelEncoder, StandardScaler
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier


# ─────────────────────────────────────────────
<<<<<<< HEAD
# ORDINAL RISK ENCODER
# Replaces LabelEncoder — preserves natural order
# ─────────────────────────────────────────────
class OrdinalRiskEncoder:
    """
    Custom Ordinal Encoder for Risk Labels.

    WHY NOT LabelEncoder?
      LabelEncoder sorts alphabetically:
        High=0, Low=1, Medium=2
      This breaks the natural order — High should be HIGHEST not 0.

    WHY Ordinal Encoding?
      Risk has a clear natural order: Low < Medium < High
      Ordinal Encoding preserves this:
        Low=0, Medium=1, High=2
      This is scientifically correct and more interpretable.

    NOTE: For tree-based models (RF, XGBoost) and Logistic Regression,
    
    label order does NOT affect model training — they treat labels as
    categorical IDs. But ordinal encoding makes the system more
    principled, interpretable, and academically correct.
    """
    def __init__(self):
        # Natural order: Low(0) < Medium(1) < High(2)
        self.classes_ = ['Low', 'Medium', 'High']
        self.mapping  = {'Low': 0, 'Medium': 1, 'High': 2}
        self.inverse  = {0: 'Low', 1: 'Medium', 2: 'High'}

    def fit_transform(self, arr):
        """Convert text labels to ordered integers."""
        return np.array([self.mapping[v] for v in arr])

    def inverse_transform(self, arr):
        """Convert ordered integers back to text labels."""
        return [self.inverse[int(i)] for i in arr]

    def transform(self, arr):
        """Transform new data using learned mapping."""
        return np.array([self.mapping[v] for v in arr])


# ─────────────────────────────────────────────
=======
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
# FEATURE COLUMNS
# (raw + engineered — must match preprocess_data.py)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # Raw sensor measurements
    'solar_wind_speed', 'proton_density', 'bx', 'by', 'bz',
    'plasma_temp', 'kp_index', 'dst_index', 'xray_flux', 'ae_index',
    'f107_index', 'ap_index', 'speed_pressure',
    # Physics-based engineered features
<<<<<<< HEAD
    'storm_severity', 'mag_disturbance','radiation_pressure',
=======
    'storm_severity', 'mag_disturbance', 'radiation_pressure',
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
    # Rolling averages
    'kp_rolling_3h', 'bz_rolling_3h', 'wind_rolling_3h', 'dst_rolling_6h',
    # Rate-of-change features
    'delta_bz', 'delta_wind', 'delta_proton', 'delta_kp',
    # Binary threshold flags
    'bz_negative', 'extreme_kp', 'extreme_dst', 'high_speed'
]


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    print("=" * 55)
    print("  SPACE WEATHER ML TRAINING (WITH PCA)")
    print("=" * 55)
    print("\n📂 Loading processed dataset...")

    if not os.path.exists('data/space_weather_processed.csv'):
        print("❌ File not found! Run preprocess_data.py first.")
        raise FileNotFoundError("data/space_weather_processed.csv not found")

    df = pd.read_csv('data/space_weather_processed.csv')
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    if len(df) == 0:
        print("❌ Dataset is empty! Run preprocess_data.py first.")
        raise ValueError("Empty dataset")

<<<<<<< HEAD
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")
=======
    # Verify all feature columns are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        raise ValueError(f"Missing columns in dataset: {missing}")
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc

    print(f"   ✅ All {len(FEATURE_COLS)} feature columns found")
    return df


# ─────────────────────────────────────────────
# 2. PREPARE X AND y
# ─────────────────────────────────────────────
def prepare_xy(df):
    print("\n🔧 Preparing features and labels...")

    X = df[FEATURE_COLS].fillna(0)
    y = df['risk_level']

<<<<<<< HEAD
    # ── ORDINAL ENCODING ──────────────────────
    # Preserves natural order: Low(0) < Medium(1) < High(2)
    # This is more correct than LabelEncoder which sorts
    # alphabetically giving: High=0, Low=1, Medium=2
    le    = OrdinalRiskEncoder()
    y_enc = le.fit_transform(y)

    print("   Encoding Type : ORDINAL (preserves natural order)")
    print("   Encoding      : Low=0  Medium=1  High=2")
    dist = dict(zip(*np.unique(y, return_counts=True)))
    print(f"   Label dist    : Low={dist.get('Low',0):,}  "
          f"Medium={dist.get('Medium',0):,}  "
          f"High={dist.get('High',0):,}")

    # ── MINMAX SCALING ──────────────────────
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   Features      : {X_scaled.shape[1]} (scaled to min=0, max=1)")
=======
    # Encode labels: High=0, Low=1, Medium=2 (alphabetical)
    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)
    print(f"   Classes   : {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"   Label dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Scale features to mean=0, std=1
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   Features  : {X_scaled.shape[1]}")
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc

    return X, X_scaled, y_enc, le, scaler


# ─────────────────────────────────────────────
# 3. APPLY PCA
# ─────────────────────────────────────────────
def apply_pca(X_scaled, y_enc, le):
    print("\n🔬 Applying PCA...")

<<<<<<< HEAD
=======
    # Full PCA to study explained variance
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_95   = int(np.argmax(cumvar >= 0.95)) + 1
    n_99   = int(np.argmax(cumvar >= 0.99)) + 1
    print(f"   Original features    : {X_scaled.shape[1]}")
    print(f"   Components @ 95% var : {n_95}")
    print(f"   Components @ 99% var : {n_99}")

<<<<<<< HEAD
    # PCA for Random Forest (95% variance)
=======
    # PCA for Logistic Regression (95% variance)
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
    pca_95   = PCA(n_components=n_95, random_state=42)
    X_pca_95 = pca_95.fit_transform(X_scaled)
    print(f"   Reduced shape        : {X_pca_95.shape}")

    # PCA 2D for scatter visualization
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d   = pca_2d.fit_transform(X_scaled)
    pd.DataFrame({
        'PC1': X_2d[:, 0], 'PC2': X_2d[:, 1],
        'risk_level': le.inverse_transform(y_enc)
    }).to_csv('data/pca_2d.csv', index=False)

    # PCA 3D for 3D scatter visualization
    pca_3d = PCA(n_components=3, random_state=42)
    X_3d   = pca_3d.fit_transform(X_scaled)
    pd.DataFrame({
        'PC1': X_3d[:, 0], 'PC2': X_3d[:, 1], 'PC3': X_3d[:, 2],
        'risk_level': le.inverse_transform(y_enc)
    }).to_csv('data/pca_3d.csv', index=False)

<<<<<<< HEAD
    # Save PCA info for dashboard charts
=======
    # Save PCA info for dashboard
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
    ev_data = {
        'explained_variance_ratio': pca_full.explained_variance_ratio_[:15].tolist(),
        'cumulative_variance'     : cumvar[:15].tolist(),
        'n_components_95'         : int(n_95),
        'n_components_99'         : int(n_99),
        'total_features'          : int(X_scaled.shape[1]),
        'loadings_pc1'            : dict(zip(FEATURE_COLS,
                                             pca_95.components_[0].tolist())),
        'loadings_pc2'            : dict(zip(FEATURE_COLS,
                                             pca_95.components_[1].tolist()))
    }
    os.makedirs('models', exist_ok=True)
    with open('models/pca_info.json', 'w') as f:
        json.dump(ev_data, f, indent=2)

    print("   ✅ PCA complete!")
    return pca_95, X_pca_95


# ─────────────────────────────────────────────
# 4. TRAIN ALL MODELS
# ─────────────────────────────────────────────
def train_models(X_scaled, X_pca, y_enc, le):
    print("\n📊 Splitting data (80% train / 20% test)...")

    X_tr,     X_te,     y_tr, y_te = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    X_pca_tr, X_pca_te, _,    _    = train_test_split(
        X_pca,    y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    print(f"   Train: {len(y_tr):,} samples  |  Test: {len(y_te):,} samples")

    results = {}
    models  = {}

<<<<<<< HEAD
    # ── Random Forest (on PCA features) ───────
    print("\n🌲 Training Random Forest (PCA features)...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_pca_tr, y_tr)
    results['Random Forest (PCA)'] = evaluate(rf.predict(X_pca_te), y_te, le)
    models['random_forest']  = rf
    print(f"   Accuracy : {results['Random Forest (PCA)']['accuracy']:.4f}")
    print(f"   F1 Score : {results['Random Forest (PCA)']['f1_score']:.4f}")

    # ── XGBoost (on PCA features) ──────────────
    print("\n⚡ Training XGBoost (PCA features)...")
=======
    # ── Random Forest ──────────────────────────
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15,
        min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    results['Random Forest'] = evaluate(rf.predict(X_te), y_te)
    models['random_forest']  = rf
    print(f"   Accuracy : {results['Random Forest']['accuracy']:.4f}")
    print(f"   F1 Score : {results['Random Forest']['f1_score']:.4f}")

    # ── XGBoost ────────────────────────────────
    print("\n⚡ Training XGBoost...")
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='mlogloss', verbosity=0)
<<<<<<< HEAD
    xgb.fit(X_pca_tr, y_tr)
    results['XGBoost (PCA)'] = evaluate(xgb.predict(X_pca_te), y_te, le)
    models['xgboost']  = xgb
    print(f"   Accuracy : {results['XGBoost (PCA)']['accuracy']:.4f}")
    print(f"   F1 Score : {results['XGBoost (PCA)']['f1_score']:.4f}")
=======
    xgb.fit(X_tr, y_tr)
    results['XGBoost'] = evaluate(xgb.predict(X_te), y_te)
    models['xgboost']  = xgb
    print(f"   Accuracy : {results['XGBoost']['accuracy']:.4f}")
    print(f"   F1 Score : {results['XGBoost']['f1_score']:.4f}")
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc

    # ── Logistic Regression (on PCA features) ──
    print("\n📈 Training Logistic Regression (PCA features)...")
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_pca_tr, y_tr)
<<<<<<< HEAD
    results['Logistic Regression (PCA)'] = evaluate(lr.predict(X_pca_te), y_te, le)
=======
    results['Logistic Regression (PCA)'] = evaluate(lr.predict(X_pca_te), y_te)
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
    models['logistic_regression']        = lr
    print(f"   Accuracy : {results['Logistic Regression (PCA)']['accuracy']:.4f}")
    print(f"   F1 Score : {results['Logistic Regression (PCA)']['f1_score']:.4f}")

    # ── Isolation Forest ───────────────────────
    print("\n🔍 Training Isolation Forest (anomaly detection)...")
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X_tr)
    anomalies = (iso.predict(X_te) == -1).sum()
    models['isolation_forest'] = iso
    print(f"   Anomalies detected in test set: {anomalies}")

<<<<<<< HEAD
    # ── Feature Importance (Explainer RF on Raw Features) ───────
    # The main models use PCA, making feature importance meaningless for raw cols.
    # To satisfy physical rule-based explanations highlighting Kp, Dst, and Bz,
    # we compute importances using a dedicated RF on core physical features.
    core_features = ['kp_index', 'dst_index', 'bz', 'solar_wind_speed', 'proton_density', 'plasma_temp', 'bx', 'by']
    core_indices = [FEATURE_COLS.index(c) for c in core_features]
    X_core_tr = X_tr[:, core_indices]
    
    rf_explainer = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_explainer.fit(X_core_tr, y_tr)
    
    feat_importance = dict(sorted(
        zip(core_features, rf_explainer.feature_importances_.tolist()),
=======
    # Feature importance from Random Forest
    feat_importance = dict(sorted(
        zip(FEATURE_COLS, rf.feature_importances_.tolist()),
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
        key=lambda x: x[1], reverse=True))

    return models, results, feat_importance


<<<<<<< HEAD
def evaluate(y_pred, y_test, le):
    """Evaluate model and include class-level report with risk labels."""
    cm = confusion_matrix(y_test, y_pred)

    # Per-class metrics with correct ordinal labels
    per_class = {}
    for i, label in enumerate(le.classes_):
        if i < len(cm):
            tp = cm[i][i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            per_class[label] = {
                'precision': round(prec, 4),
                'recall'   : round(rec,  4),
                'f1'       : round(f1,   4)
            }

    return {
        'accuracy'     : round(accuracy_score(y_test, y_pred), 4),
        'precision'    : round(precision_score(y_test, y_pred,
                               average='weighted', zero_division=0), 4),
        'recall'       : round(recall_score(y_test, y_pred,
                               average='weighted', zero_division=0), 4),
        'f1_score'     : round(f1_score(y_test, y_pred,
                               average='weighted', zero_division=0), 4),
        'confusion_matrix': cm.tolist(),
        'per_class'    : per_class
=======
def evaluate(y_pred, y_test):
    return {
        'accuracy' : round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred,
                           average='weighted', zero_division=0), 4),
        'recall'   : round(recall_score(y_test, y_pred,
                           average='weighted', zero_division=0), 4),
        'f1_score' : round(f1_score(y_test, y_pred,
                           average='weighted', zero_division=0), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
    }


# ─────────────────────────────────────────────
# 5. SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(models, pca_95, scaler, le, results, feat_importance):
    os.makedirs('models', exist_ok=True)
    print("\n💾 Saving models and artifacts...")

    for name, model in models.items():
        with open(f'models/{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"   Saved: models/{name}.pkl")

    with open('models/pca.pkl', 'wb') as f:
        pickle.dump(pca_95, f)
    print("   Saved: models/pca.pkl")

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("   Saved: models/scaler.pkl")

<<<<<<< HEAD
    # Save OrdinalRiskEncoder (replaces LabelEncoder)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("   Saved: models/label_encoder.pkl  (OrdinalRiskEncoder)")
=======
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("   Saved: models/label_encoder.pkl")
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc

    with open('models/feature_cols.pkl', 'wb') as f:
        pickle.dump(FEATURE_COLS, f)
    print("   Saved: models/feature_cols.pkl")

    with open('models/results.json', 'w') as f:
        json.dump({
            'model_results'     : results,
            'feature_importance': feat_importance,
<<<<<<< HEAD
            'classes'           : le.classes_,
            'encoding'          : {'Low': 0, 'Medium': 1, 'High': 2}
=======
            'classes'           : le.classes_.tolist()
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
        }, f, indent=2)
    print("   Saved: models/results.json")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df                                  = load_data()
    X, X_scaled, y_enc, le, scaler      = prepare_xy(df)
    pca_95, X_pca                       = apply_pca(X_scaled, y_enc, le)
    models, results, feat_imp           = train_models(X_scaled, X_pca, y_enc, le)
    save_artifacts(models, pca_95, scaler, le, results, feat_imp)

    print("\n" + "=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)
    for name, r in results.items():
        print(f"  {name:35s} | Acc: {r['accuracy']:.4f} | F1: {r['f1_score']:.4f}")

<<<<<<< HEAD
    print("\n📊 Per-Class Performance (XGBoost):")
    for label, metrics in results['XGBoost (PCA)']['per_class'].items():
        print(f"   {label:8s} | Precision: {metrics['precision']:.4f} "
              f"| Recall: {metrics['recall']:.4f} "
              f"| F1: {metrics['f1']:.4f}")

    print("\n✅ TRAINING COMPLETE — Ordinal Encoding Applied")
    print("   Encoding: Low=0  Medium=1  High=2")
=======
    print("\n✅ TRAINING COMPLETE")
>>>>>>> c5010ad008495405d64a86bdda8c3e2a17da5bdc
    print("   Run: streamlit run dashboard/app.py")
