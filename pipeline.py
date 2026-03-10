#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end pipeline: Veri üret → Model eğit → v2 eşleştir → v1 vs v2 karşılaştır.

Kullanım:
    python pipeline.py
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def step1_generate_feedback():
    """Adım 1: 5 yıllık sentetik geri bildirim verisi üret."""
    print("=" * 60)
    print("ADIM 1: 5 Yıllık Sentetik Geri Bildirim Üretimi")
    print("=" * 60)

    from generate_feedback import generate_feedback
    df = generate_feedback(
        rooms_path="rooms.csv",
        n_students_per_year=50,
        n_years=5,
        output_path="feedback_5years.csv",
        seed=42,
    )
    print(f"\nÖzet: {len(df)} kayıt, {df['year'].nunique()} yıl")
    print(f"Memnuniyet ortalaması: {df['satisfaction'].mean():.4f}")
    print(f"Memnuniyet std: {df['satisfaction'].std():.4f}")
    return df


def step2_train_model():
    """Adım 2: Random Forest modelini eğit."""
    print("\n" + "=" * 60)
    print("ADIM 2: Random Forest Model Eğitimi")
    print("=" * 60)

    from ai_model import train_model, get_feature_importance, save_model

    model, metrics = train_model(
        feedback_path="feedback_5years.csv",
        n_estimators=200,
        cv_folds=5,
    )

    # Feature importance raporu
    print("\n--- Özellik Önem Sıralaması ---")
    fi = get_feature_importance(model)
    for _, row in fi.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:40s} {row['importance']:.4f}  {bar}")

    save_model(model)
    return model, metrics


def step3_v2_matching(alpha=0.7):
    """Adım 3: v2 ile yeni öğrencileri eşleştir."""
    print("\n" + "=" * 60)
    print(f"ADIM 3: AI Destekli Eşleştirme (v2, alpha={alpha})")
    print("=" * 60)

    from kyk_matcher_v2 import main as v2_main

    assignments_v2, flow_v2, score_v2 = v2_main(
        rooms_path="rooms.csv",
        new_students_path="new_students.csv",
        model_path="trained_model.joblib",
        output_path="assignments_v2.csv",
        alpha=alpha,
    )
    return assignments_v2, flow_v2, score_v2


def step4_compare():
    """Adım 4: v1 vs v2 karşılaştırma."""
    print("\n" + "=" * 60)
    print("ADIM 4: v1 vs v2 Karşılaştırma")
    print("=" * 60)

    from kyk_matcher_v1 import main as v1_main

    # v1'i tekrar çalıştır (güncel karşılaştırma için)
    print("\n--- v1 (Kural Tabanlı) ---")
    v1_main(
        rooms_path="rooms.csv",
        new_students_path="new_students.csv",
        output_path="assignments_v1_compare.csv",
    )

    v1 = pd.read_csv(os.path.join(BASE_DIR, "assignments_v1_compare.csv"))
    v2 = pd.read_csv(os.path.join(BASE_DIR, "assignments_v2.csv"))

    print("\n" + "=" * 60)
    print("KARŞILAŞTIRMA RAPORU")
    print("=" * 60)

    # Atanan sayıları
    v1_assigned = (v1["status"] == "assigned").sum()
    v2_assigned = (v2["status"] == "assigned").sum()
    print(f"\nAtanan öğrenci:  v1={v1_assigned}, v2={v2_assigned}")

    # Ortalama skorlar
    v1_scores = v1[v1["status"] == "assigned"]["score"]
    v2_scores = v2[v2["status"] == "assigned"]["score"]
    print(f"Ortalama skor:   v1={v1_scores.mean():.4f}, v2={v2_scores.mean():.4f}")
    print(f"Minimum skor:    v1={v1_scores.min():.4f}, v2={v2_scores.min():.4f}")
    print(f"Maksimum skor:   v1={v1_scores.max():.4f}, v2={v2_scores.max():.4f}")
    print(f"Toplam skor:     v1={v1_scores.sum():.4f}, v2={v2_scores.sum():.4f}")

    # Detaylı öğrenci bazlı karşılaştırma
    print("\n--- Öğrenci Bazlı Karşılaştırma ---")
    print(f"{'Öğrenci':<20} {'v1 Oda':<10} {'v1 Skor':<10} {'v2 Oda':<10} {'v2 Skor':<10} {'Fark':<10}")
    print("-" * 70)

    merged = v1.merge(v2, on="student_name", suffixes=("_v1", "_v2"))
    for _, row in merged.iterrows():
        name = row["student_name"]
        r1 = row.get("assigned_room_id_v1", "N/A")
        s1 = row.get("score_v1", 0)
        r2 = row.get("assigned_room_id_v2", "N/A")
        s2 = row.get("score_v2", 0)

        s1 = float(s1) if pd.notna(s1) else 0
        s2 = float(s2) if pd.notna(s2) else 0
        diff = s2 - s1
        indicator = "▲" if diff > 0 else ("▼" if diff < 0 else "=")

        print(f"{name:<20} {str(r1):<10} {s1:<10.4f} {str(r2):<10} {s2:<10.4f} {diff:+.4f} {indicator}")

    # Temizlik
    compare_path = os.path.join(BASE_DIR, "assignments_v1_compare.csv")
    if os.path.exists(compare_path):
        os.remove(compare_path)


def main():
    """Tüm pipeline'ı çalıştır."""
    print("\n" + "+" + "=" * 58 + "+")
    print("|" + " KYK ODA ESLESTIRME - AI PIPELINE ".center(58) + "|")
    print("+" + "=" * 58 + "+\n")

    step1_generate_feedback()
    model, metrics = step2_train_model()
    assignments_v2, flow_v2, score_v2 = step3_v2_matching(alpha=0.7)
    step4_compare()

    print("\n" + "+" + "=" * 58 + "+")
    print("|" + " PIPELINE TAMAMLANDI ".center(58) + "|")
    print("+" + "=" * 58 + "+")
    print("\nÜretilen dosyalar:")
    print("  - feedback_5years.csv     : 5 yillik egitim verisi")
    print("  - trained_model.joblib    : Egitilmis Random Forest modeli")
    print("  - assignments_v2.csv      : AI destekli eslestirme sonuclari")


if __name__ == "__main__":
    main()
