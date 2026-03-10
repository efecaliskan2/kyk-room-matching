#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI destekli oda eşleştirme sistemi (v2).

Random Forest modeli ile öğrenci-oda uyum skorunu tahmin eder,
ardından Min-Cost Max-Flow ile optimal atama yapar.
v1'in hard constraint'leri ve akış çözücüsü aynen korunur.
"""

import os
import pandas as pd
import numpy as np

from kyk_matcher_v1 import (
    Columns, Weights, Hard, TimeCfg, ScoringCfg,
    MinCostMaxFlow, build_room_slots,
    violates_hard_constraints, compute_soft_score, clamp_1_10
)
from ai_model import predict_score, ALL_FEATURES, STUDENT_FEATURES, ROOM_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_ai_score(model, student_row, room_row, C):
    """
    AI modeli ile öğrenci-oda uyum skorunu tahmin eder.
    """
    s_feats = [
        int(student_row[C.noise_tolerance]),
        int(student_row[C.smoking_level]),
        int(student_row[C.environment_sensitivity]),
        int(student_row[C.wake_time]),
        int(student_row[C.entry_time]),
        int(student_row[C.sleep_interrupt_sensitivity]),
    ]
    r_feats = [
        int(room_row[C.room_noise]),
        int(room_row[C.room_smoking]),
        int(room_row[C.room_env]),
        int(room_row[C.room_wake]),
        int(room_row[C.room_entry]),
    ]
    return predict_score(model, s_feats, r_feats)


def compute_hybrid_score(model, student_row, room_row, C, W, T, alpha=0.7):
    """
    Hibrit skor: alpha * AI_score + (1-alpha) * rule_score

    alpha = 1.0 → tam AI tabanlı
    alpha = 0.0 → tam kural tabanlı (v1)
    alpha = 0.7 → varsayılan hibrit (%70 AI, %30 kural)
    """
    ai_score = compute_ai_score(model, student_row, room_row, C)
    rule_score = compute_soft_score(student_row, room_row, C, W, T)
    return alpha * ai_score + (1 - alpha) * rule_score


def batch_assign_v2(
    students, rooms, model, C, W, H, T, S,
    alpha=0.7, use_hybrid=True
):
    """
    AI destekli toplu öğrenci-oda ataması.

    v1'den farkı: compute_soft_score yerine AI/hibrit skor kullanır.
    Geri kalan akış (hard constraint, MCMF) aynıdır.

    Args:
        students: Yeni öğrenci DataFrame
        rooms: Oda DataFrame
        model: Eğitilmiş RandomForest modeli
        alpha: Hibrit modda AI ağırlığı
        use_hybrid: True ise hibrit, False ise sadece AI skoru

    Returns:
        (assignments_df, flow, total_score)
    """
    slots = build_room_slots(rooms, C)
    if len(slots) == 0:
        out = students[[C.student_id]].copy()
        out["assigned_room_id"] = None
        out["score"] = None
        out["status"] = "unassigned_no_capacity"
        return out, 0, 0

    n_students = len(students)
    n_slots = len(slots)

    SRC = 0
    STUD_START = 1
    SLOT_START = STUD_START + n_students
    SNK = SLOT_START + n_slots

    mcmf = MinCostMaxFlow(SNK + 1)

    for i in range(n_students):
        mcmf.add_edge(SRC, STUD_START + i, 1, 0)
    for j in range(n_slots):
        mcmf.add_edge(SLOT_START + j, SNK, 1, 0)

    room_by_id = rooms.set_index(C.room_id)

    feasible_edge_count = 0
    for i in range(n_students):
        srow = students.iloc[i]
        for j in range(n_slots):
            room_id = slots[j][1]
            rrow = room_by_id.loc[room_id]

            # Hard constraint'ler aynen korunuyor
            if violates_hard_constraints(srow, rrow, C, H):
                continue

            # Skor hesaplama: AI veya hibrit
            if use_hybrid:
                score = compute_hybrid_score(model, srow, rrow, C, W, T, alpha)
            else:
                score = compute_ai_score(model, srow, rrow, C)

            if score < S.min_score_to_consider:
                continue

            cost = -int(round(score * S.score_scale))
            mcmf.add_edge(STUD_START + i, SLOT_START + j, 1, cost)
            feasible_edge_count += 1

    if feasible_edge_count == 0:
        out = students[[C.student_id]].copy()
        out["assigned_room_id"] = None
        out["score"] = None
        out["status"] = "unassigned_no_feasible"
        return out, 0, 0

    maxf = min(n_students, n_slots)
    flow, cost = mcmf.min_cost_flow(SRC, SNK, maxf)

    assigned_room = [None] * n_students
    assigned_score = [None] * n_students

    for i in range(n_students):
        v = STUD_START + i
        for e in mcmf.g[v]:
            if SLOT_START <= e.to < SNK and e.cap == 0:
                slot_idx = e.to - SLOT_START
                room_id = slots[slot_idx][1]
                assigned_room[i] = room_id
                assigned_score[i] = (-e.cost) / S.score_scale
                break

    feasible_for_student = [False] * n_students
    for i in range(n_students):
        v = STUD_START + i
        for e in mcmf.g[v]:
            if SLOT_START <= e.to < SNK:
                feasible_for_student[i] = True
                break

    out = students[[C.student_id]].copy()
    out["assigned_room_id"] = assigned_room
    out["score"] = assigned_score

    status = []
    for i in range(n_students):
        if assigned_room[i] is not None:
            status.append("assigned")
        else:
            status.append(
                "unassigned_no_capacity" if feasible_for_student[i]
                else "unassigned_no_feasible"
            )
    out["status"] = status

    total_score = (-cost) / S.score_scale
    return out, flow, total_score


def main(
    rooms_path="rooms.csv",
    new_students_path="new_students.csv",
    model_path="trained_model.joblib",
    output_path="assignments_v2.csv",
    alpha=0.7,
):
    """Ana v2 eşleştirme fonksiyonu."""
    from ai_model import load_model

    C = Columns()
    W = Weights()
    H = Hard()
    T = TimeCfg()
    S = ScoringCfg()

    rooms = pd.read_csv(os.path.join(BASE_DIR, rooms_path))
    students = pd.read_csv(os.path.join(BASE_DIR, new_students_path))

    if C.current_occupancy not in rooms.columns:
        rooms[C.current_occupancy] = 0

    model = load_model(model_path)

    assignments, flow, total_score = batch_assign_v2(
        students, rooms, model, C, W, H, T, S,
        alpha=alpha, use_hybrid=True
    )
    assignments.to_csv(os.path.join(BASE_DIR, output_path), index=False)

    assigned_count = int((assignments["status"] == "assigned").sum())
    print(f"Kaydedildi: {output_path}")
    print(f"Atanan öğrenci sayısı: {assigned_count}")
    print(f"Akış (flow): {flow}")
    print(f"Toplam uyum skoru: {total_score:.3f}")
    print(f"Hibrit alpha: {alpha} (AI: {alpha*100:.0f}%, Kural: {(1-alpha)*100:.0f}%)")
    print(assignments["status"].value_counts(dropna=False))

    return assignments, flow, total_score


if __name__ == "__main__":
    main()
