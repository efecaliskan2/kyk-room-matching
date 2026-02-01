#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import pandas as pd
from dataclasses import dataclass
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Columns:
    student_id: str = "student_name"
    room_id: str = "room_id"

    noise_tolerance: str = "noise_tolerance"
    smoking_level: str = "smoking_level"
    environment_sensitivity: str = "environment_sensitivity"
    wake_time: str = "wake_time"
    entry_time: str = "entry_time"
    sleep_interrupt_sensitivity: str = "sleep_interrupt_sensitivity"

    room_noise: str = "room_noise_profile"
    room_smoking: str = "room_smoking_profile"
    room_env: str = "room_environment_irritant_level"
    room_wake: str = "room_wake_profile"
    room_entry: str = "room_entry_profile"

    capacity: str = "capacity"
    current_occupancy: str = "current_occupancy"

@dataclass
class Weights:
    noise: float = 2.0
    smoking: float = 2.5
    env: float = 2.0
    wake: float = 1.2
    entry: float = 1.0
    sleep_interrupt: float = 2.0

@dataclass
class Hard:
    student_smoking_low: int = 2
    room_smoking_high: int = 8
    student_env_high: int = 8
    room_env_high: int = 8

@dataclass
class TimeCfg:
    period_wake: int = 10
    period_entry: int = 10

@dataclass
class ScoringCfg:
    score_scale: int = 1000
    min_score_to_consider: float = 0.25

def clamp_1_10(x):
    x = float(x)
    if x < 1.0:
        return 1.0
    if x > 10.0:
        return 10.0
    return x

def linear_similarity(a, b):
    a = clamp_1_10(a)
    b = clamp_1_10(b)
    diff = abs(a - b)
    return 1.0 - (diff / 9.0)

def circular_distance(a, b, period):
    a0 = (int(a) - 1) % period
    b0 = (int(b) - 1) % period
    d = abs(a0 - b0)
    return min(d, period - d)

def circular_similarity(a, b, period):
    d = circular_distance(a, b, period)
    max_d = period // 2
    if max_d <= 0:
        return 1.0
    return 1.0 - (d / max_d)

def violates_hard_constraints(s, r, C, H):
    if int(s[C.smoking_level]) <= H.student_smoking_low and int(r[C.room_smoking]) >= H.room_smoking_high:
        return True
    if int(s[C.environment_sensitivity]) >= H.student_env_high and int(r[C.room_env]) >= H.room_env_high:
        return True
    return False

def compute_soft_score(s, r, C, W, T):
    sim_noise = linear_similarity(s[C.noise_tolerance], r[C.room_noise])
    sim_smoking = linear_similarity(s[C.smoking_level], r[C.room_smoking])
    sim_env = linear_similarity(s[C.environment_sensitivity], r[C.room_env])

    sim_wake = circular_similarity(int(s[C.wake_time]), int(r[C.room_wake]), T.period_wake)
    sim_entry = circular_similarity(int(s[C.entry_time]), int(r[C.room_entry]), T.period_entry)

    desired_quiet = 11 - clamp_1_10(r[C.room_noise])
    sim_sleep = linear_similarity(s[C.sleep_interrupt_sensitivity], desired_quiet)

    total_w = W.noise + W.smoking + W.env + W.wake + W.entry + W.sleep_interrupt
    score = (
        W.noise * sim_noise +
        W.smoking * sim_smoking +
        W.env * sim_env +
        W.wake * sim_wake +
        W.entry * sim_entry +
        W.sleep_interrupt * sim_sleep
    ) / total_w
    return float(score)

class Edge:
    __slots__ = ("to", "rev", "cap", "cost")
    def __init__(self, to, rev, cap, cost):
        self.to = to
        self.rev = rev
        self.cap = cap
        self.cost = cost

class MinCostMaxFlow:
    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, fr, to, cap, cost):
        fwd = Edge(to, len(self.g[to]), cap, cost)
        rev = Edge(fr, len(self.g[fr]), 0, -cost)
        self.g[fr].append(fwd)
        self.g[to].append(rev)

    def min_cost_flow(self, s, t, maxf):
        n = self.n
        INF = 10**18
        res_cost = 0
        flow = 0

        h = [0] * n
        prevv = [0] * n
        preve = [0] * n

        import heapq

        while flow < maxf:
            dist = [INF] * n
            dist[s] = 0
            pq = [(0, s)]

            while pq:
                d, v = heapq.heappop(pq)
                if dist[v] < d:
                    continue
                for i, e in enumerate(self.g[v]):
                    if e.cap <= 0:
                        continue
                    nd = d + e.cost + h[v] - h[e.to]
                    if nd < dist[e.to]:
                        dist[e.to] = nd
                        prevv[e.to] = v
                        preve[e.to] = i
                        heapq.heappush(pq, (nd, e.to))

            if dist[t] == INF:
                break

            for v in range(n):
                if dist[v] < INF:
                    h[v] += dist[v]

            d = maxf - flow
            v = t
            while v != s:
                d = min(d, self.g[prevv[v]][preve[v]].cap)
                v = prevv[v]

            v = t
            while v != s:
                e = self.g[prevv[v]][preve[v]]
                e.cap -= d
                self.g[v][e.rev].cap += d
                v = prevv[v]

            flow += d
            res_cost += d * h[t]

        return flow, res_cost

def build_room_slots(rooms, C):
    slots = []
    for _, r in rooms.iterrows():
        cap = int(r[C.capacity])
        occ = int(r[C.current_occupancy]) if C.current_occupancy in rooms.columns else 0
        free = max(0, cap - occ)
        for k in range(free):
            slots.append((str(r[C.room_id]) + "__slot" + str(k + 1), str(r[C.room_id])))
    return slots

def batch_assign(students, rooms, C, W, H, T, S):
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

            if violates_hard_constraints(srow, rrow, C, H):
                continue

            score = compute_soft_score(srow, rrow, C, W, T)
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
            status.append("unassigned_no_capacity" if feasible_for_student[i] else "unassigned_no_feasible")
    out["status"] = status

    total_score = (-cost) / S.score_scale
    return out, flow, total_score

def main(rooms_path="rooms.csv", new_students_path="new_students.csv", output_path="assignments.csv"):
    C = Columns()
    W = Weights()
    H = Hard()
    T = TimeCfg()
    S = ScoringCfg()

    rooms = pd.read_csv(os.path.join(BASE_DIR, rooms_path))
    students = pd.read_csv(os.path.join(BASE_DIR, new_students_path))


    required_rooms = [C.room_id, C.capacity, C.room_noise, C.room_smoking, C.room_env, C.room_wake, C.room_entry]
    required_students = [C.student_id, C.noise_tolerance, C.smoking_level, C.environment_sensitivity, C.wake_time, C.entry_time, C.sleep_interrupt_sensitivity]

    missing_r = [c for c in required_rooms if c not in rooms.columns]
    missing_s = [c for c in required_students if c not in students.columns]
    if missing_r:
        raise ValueError("rooms.csv eksik kolonlar: " + str(missing_r))
    if missing_s:
        raise ValueError("new_students.csv eksik kolonlar: " + str(missing_s))

    if C.current_occupancy not in rooms.columns:
        rooms[C.current_occupancy] = 0

    assignments, flow, total_score = batch_assign(students, rooms, C, W, H, T, S)
    assignments.to_csv(output_path, index=False)

    assigned_count = int((assignments["status"] == "assigned").sum())
    print("Kaydedildi:", output_path)
    print("Atanan ogrenci sayisi:", assigned_count)
    print("Akis (flow):", flow)
    print("Toplam uyum skoru:", total_score)
    print(assignments["status"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
