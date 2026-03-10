#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest tabanlı uyum tahmin modeli.

5 yıllık eşleştirme + memnuniyet verisinden öğrenerek,
öğrenci-oda çiftleri için uyum skoru tahmin eder.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model giriş özellikleri (sıra önemli!)
STUDENT_FEATURES = [
    "noise_tolerance",
    "smoking_level",
    "environment_sensitivity",
    "wake_time",
    "entry_time",
    "sleep_interrupt_sensitivity",
]

ROOM_FEATURES = [
    "room_noise_profile",
    "room_smoking_profile",
    "room_environment_irritant_level",
    "room_wake_profile",
    "room_entry_profile",
]

ALL_FEATURES = STUDENT_FEATURES + ROOM_FEATURES
TARGET_COL = "satisfaction"


def prepare_features(df):
    """DataFrame'den özellik matrisi (X) ve hedef vektör (y) çıkar."""
    X = df[ALL_FEATURES].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(np.float64) if TARGET_COL in df.columns else None
    return X, y


def train_model(
    feedback_path="feedback_5years.csv",
    n_estimators=200,
    max_depth=None,
    random_state=42,
    cv_folds=5,
):
    """
    Random Forest Regressor eğitir.

    Args:
        feedback_path: Eğitim verisi CSV dosyası
        n_estimators: Ağaç sayısı
        max_depth: Maksimum derinlik (None = sınırsız)
        random_state: Tekrarlanabilirlik için seed
        cv_folds: Cross-validation katman sayısı

    Returns:
        (model, metrics_dict)
    """
    df = pd.read_csv(os.path.join(BASE_DIR, feedback_path))
    X, y = prepare_features(df)

    print(f"Eğitim verisi: {X.shape[0]} örnek, {X.shape[1]} özellik")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,  # tüm CPU çekirdeklerini kullan
    )

    # Cross-validation
    cv_scores = cross_val_score(
        model, X, y,
        cv=cv_folds,
        scoring="r2",
    )

    # Tam eğitim
    model.fit(X, y)
    y_pred = model.predict(X)

    metrics = {
        "r2_train": r2_score(y, y_pred),
        "rmse_train": np.sqrt(mean_squared_error(y, y_pred)),
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "n_samples": len(y),
    }

    print(f"\n--- Model Performansı ---")
    print(f"R² (eğitim):          {metrics['r2_train']:.4f}")
    print(f"RMSE (eğitim):        {metrics['rmse_train']:.4f}")
    print(f"R² (CV {cv_folds}-fold):     {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")

    return model, metrics


def get_feature_importance(model):
    """
    Özellik önem sıralamasını döndürür.

    Returns:
        DataFrame: (feature, importance) sıralı
    """
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def predict_score(model, student_features, room_features):
    """
    Tek bir öğrenci-oda çifti için uyum skoru tahmin eder.

    Args:
        model: Eğitilmiş RandomForestRegressor
        student_features: [noise_tol, smoking, env_sens, wake, entry, sleep_int]
        room_features: [room_noise, room_smoking, room_env, room_wake, room_entry]

    Returns:
        float: Tahmin edilen uyum skoru (0-1 arası)
    """
    x = np.array(student_features + room_features, dtype=np.float64).reshape(1, -1)
    score = model.predict(x)[0]
    return float(np.clip(score, 0.0, 1.0))


def predict_batch(model, students_df, rooms_df):
    """
    Tüm öğrenci-oda çiftleri için tahmin matrisi oluşturur.

    Returns:
        dict: {(student_idx, room_id): score}
    """
    from kyk_matcher_v1 import Columns
    C = Columns()

    scores = {}
    room_by_id = rooms_df.set_index(C.room_id)

    for i, srow in students_df.iterrows():
        s_feats = [
            int(srow[C.noise_tolerance]),
            int(srow[C.smoking_level]),
            int(srow[C.environment_sensitivity]),
            int(srow[C.wake_time]),
            int(srow[C.entry_time]),
            int(srow[C.sleep_interrupt_sensitivity]),
        ]
        for room_id, rrow in room_by_id.iterrows():
            r_feats = [
                int(rrow[C.room_noise]),
                int(rrow[C.room_smoking]),
                int(rrow[C.room_env]),
                int(rrow[C.room_wake]),
                int(rrow[C.room_entry]),
            ]
            scores[(i, room_id)] = predict_score(model, s_feats, r_feats)

    return scores


def save_model(model, path="trained_model.joblib"):
    """Modeli diske kaydet."""
    full_path = os.path.join(BASE_DIR, path)
    joblib.dump(model, full_path)
    print(f"Model kaydedildi: {path}")


def load_model(path="trained_model.joblib"):
    """Modeli diskten yükle."""
    full_path = os.path.join(BASE_DIR, path)
    model = joblib.load(full_path)
    print(f"Model yüklendi: {path}")
    return model


if __name__ == "__main__":
    print("=== Random Forest Model Eğitimi ===\n")

    model, metrics = train_model()

    print("\n--- Özellik Önem Sıralaması ---")
    fi = get_feature_importance(model)
    for _, row in fi.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:40s} {row['importance']:.4f}  {bar}")

    save_model(model)
