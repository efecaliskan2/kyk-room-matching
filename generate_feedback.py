#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5 yıllık sentetik eşleştirme + memnuniyet geri bildirimi üretici.

Gerçek senaryoda memnuniyet verileri anket/form ile toplanır.
Bu modül, v1 algoritmasını kullanarak 5 yıllık simülasyon verisini
otomatik olarak üretir — AI modelinin eğitilmesi için.
"""

import os
import random
import numpy as np
import pandas as pd

from kyk_matcher_v1 import (
    Columns, Weights, Hard, TimeCfg, ScoringCfg,
    batch_assign, compute_soft_score, violates_hard_constraints
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def random_students(n, seed=None):
    """n adet rastgele öğrenci üret."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    C = Columns()
    rows = []
    for i in range(n):
        rows.append({
            C.student_id: f"Ogrenci_Y{{}}_{i+1}",  # yıl sonra doldurulacak
            C.noise_tolerance: random.randint(1, 10),
            C.smoking_level: random.randint(1, 10),
            C.environment_sensitivity: random.randint(1, 10),
            C.wake_time: random.randint(1, 10),
            C.entry_time: random.randint(1, 10),
            C.sleep_interrupt_sensitivity: random.randint(1, 10),
        })
    return pd.DataFrame(rows)


def simulate_satisfaction(algo_score, noise_std=0.08):
    """
    Algoritma skorundan memnuniyet skoru üretir.
    Gerçek senaryoda anketlerden gelir; burada simüle ediyoruz.
    Algo skoru ile korelasyonlu ama gürültülü — 
    çünkü gerçekte insanlar algoritmadan biraz farklı tepki verir.
    """
    noise = np.random.normal(0, noise_std)
    satisfaction = algo_score + noise
    return float(np.clip(satisfaction, 0.0, 1.0))


def generate_feedback(
    rooms_path="rooms.csv",
    n_students_per_year=50,
    n_years=5,
    output_path="feedback_5years.csv",
    seed=42
):
    """
    5 yıllık sentetik eşleştirme + memnuniyet verisini üretir.

    Her yıl:
    1. Rastgele öğrenci havuzu oluştur
    2. v1 algoritması ile eşleştir
    3. Eşleştirme skorundan memnuniyet skoru simüle et
    4. Tüm özellikleri (öğrenci + oda) kaydet

    Returns:
        DataFrame: Tüm yılların birleşik eğitim verisi
    """
    C = Columns()
    W = Weights()
    H = Hard()
    T = TimeCfg()
    S = ScoringCfg()

    rooms = pd.read_csv(os.path.join(BASE_DIR, rooms_path))
    if C.current_occupancy not in rooms.columns:
        rooms[C.current_occupancy] = 0

    room_by_id = rooms.set_index(C.room_id)

    all_records = []

    for year in range(1, n_years + 1):
        year_seed = seed + year * 1000
        students = random_students(n_students_per_year, seed=year_seed)

        # Öğrenci isimlerini yıla göre güncelle
        students[C.student_id] = [
            f"Ogrenci_Y{year}_{i+1}" for i in range(len(students))
        ]

        # v1 ile eşleştir
        assignments, flow, total_score = batch_assign(
            students, rooms, C, W, H, T, S
        )

        # Her atanan öğrenci için kayıt oluştur
        for idx, arow in assignments.iterrows():
            if arow["status"] != "assigned":
                continue

            room_id = arow["assigned_room_id"]
            algo_score = float(arow["score"])
            student_row = students[students[C.student_id] == arow[C.student_id]].iloc[0]
            room_row = room_by_id.loc[room_id]

            satisfaction = simulate_satisfaction(algo_score)

            record = {
                "year": year,
                "student_name": arow[C.student_id],
                "room_id": room_id,
                # Öğrenci özellikleri
                "noise_tolerance": int(student_row[C.noise_tolerance]),
                "smoking_level": int(student_row[C.smoking_level]),
                "environment_sensitivity": int(student_row[C.environment_sensitivity]),
                "wake_time": int(student_row[C.wake_time]),
                "entry_time": int(student_row[C.entry_time]),
                "sleep_interrupt_sensitivity": int(student_row[C.sleep_interrupt_sensitivity]),
                # Oda özellikleri
                "room_noise_profile": int(room_row[C.room_noise]),
                "room_smoking_profile": int(room_row[C.room_smoking]),
                "room_environment_irritant_level": int(room_row[C.room_env]),
                "room_wake_profile": int(room_row[C.room_wake]),
                "room_entry_profile": int(room_row[C.room_entry]),
                # Skorlar
                "algo_score": round(algo_score, 4),
                "satisfaction": round(satisfaction, 4),
            }
            all_records.append(record)

        print(f"  Yıl {year}: {flow} öğrenci eşleştirildi")

    df = pd.DataFrame(all_records)
    output_full = os.path.join(BASE_DIR, output_path)
    df.to_csv(output_full, index=False)
    print(f"\nToplam {len(df)} kayıt -> {output_path}")
    return df


if __name__ == "__main__":
    print("=== 5 Yıllık Sentetik Geri Bildirim Üretimi ===\n")
    generate_feedback()
