#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KYK Oda Eslestirme - AI Destekli Arayuz (v2)

Bu arayuz:
1. Oda CSV ve ogrenci CSV yukler
2. 5 yillik sentetik veri uretir (veya mevcut feedback CSV yukler)
3. Random Forest modelini egitir
4. AI destekli eslestirme yapar
5. v1 vs v2 karsilastirma gosterir
6. Sonuclari CSV olarak kaydeder
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

import pandas as pd
import numpy as np

from kyk_matcher_v1 import (
    Columns, Weights, Hard, TimeCfg, ScoringCfg,
    batch_assign, compute_soft_score
)
from ai_model import (
    train_model, get_feature_importance,
    save_model, load_model, predict_score
)
from kyk_matcher_v2 import batch_assign_v2
from generate_feedback import generate_feedback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class KykMatcherV2App:
    def __init__(self, root):
        self.root = root
        self.root.title("KYK Oda Eslestirme - AI Destekli (v2)")
        self.root.geometry("1200x820")
        self.root.configure(bg="#1a1a2e")
        self.root.minsize(1000, 700)

        self.rooms_df = None
        self.students_df = None
        self.feedback_df = None
        self.model = None
        self.results_v1 = None
        self.results_v2 = None

        self.alpha_var = tk.DoubleVar(value=0.7)

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Dark.TFrame", background="#1a1a2e")
        style.configure("Card.TFrame", background="#16213e")
        style.configure("Dark.TLabel", background="#1a1a2e", foreground="#e0e0e0")
        style.configure("Title.TLabel", background="#1a1a2e", foreground="#e94560",
                         font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background="#1a1a2e", foreground="#a0a0b0",
                         font=("Segoe UI", 10))
        style.configure("Metric.TLabel", background="#16213e", foreground="#00d4aa",
                         font=("Consolas", 11, "bold"))
        style.configure("Status.TLabel", background="#0f3460", foreground="#e0e0e0",
                         font=("Segoe UI", 9))

        # Treeview stilleri
        style.configure("Custom.Treeview",
                         background="#16213e",
                         foreground="#e0e0e0",
                         fieldbackground="#16213e",
                         font=("Consolas", 9))
        style.configure("Custom.Treeview.Heading",
                         background="#0f3460",
                         foreground="#e0e0e0",
                         font=("Segoe UI", 9, "bold"))

        # Ana cerceve
        main = tk.Frame(self.root, bg="#1a1a2e", padx=16, pady=12)
        main.pack(fill=tk.BOTH, expand=True)

        # Baslik
        title_frame = tk.Frame(main, bg="#1a1a2e")
        title_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(title_frame, text="KYK Oda Eslestirme",
                 font=("Segoe UI", 20, "bold"), bg="#1a1a2e",
                 fg="#e94560").pack(side=tk.LEFT)
        tk.Label(title_frame, text="  AI Destekli v2",
                 font=("Segoe UI", 14), bg="#1a1a2e",
                 fg="#a0a0b0").pack(side=tk.LEFT, pady=(6, 0))

        # Buton satiri
        btn_frame = tk.Frame(main, bg="#1a1a2e")
        btn_frame.pack(fill=tk.X, pady=6)

        buttons = [
            ("Oda CSV Yukle", self.load_rooms, "#2ecc71"),
            ("Ogrenci CSV Yukle", self.load_students, "#3498db"),
            ("Feedback Uret / Yukle", self.handle_feedback, "#9b59b6"),
            ("Model Egit", self.train_ai_model, "#e67e22"),
            ("v2 Eslestir", self.run_v2_matching, "#e94560"),
            ("Sonuclari Kaydet", self.save_results, "#16a085"),
        ]

        for text, cmd, color in buttons:
            tk.Button(
                btn_frame, text=text, command=cmd,
                bg=color, fg="white", font=("Segoe UI", 9, "bold"),
                padx=10, pady=6, cursor="hand2",
                activebackground=color, activeforeground="white",
                relief=tk.FLAT, bd=0
            ).pack(side=tk.LEFT, padx=3)

        # Alpha slider
        alpha_frame = tk.Frame(main, bg="#1a1a2e")
        alpha_frame.pack(fill=tk.X, pady=4)

        tk.Label(alpha_frame, text="AI Agirligi (alpha):",
                 font=("Segoe UI", 9), bg="#1a1a2e",
                 fg="#a0a0b0").pack(side=tk.LEFT, padx=(0, 6))

        self.alpha_scale = tk.Scale(
            alpha_frame, from_=0.0, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL, variable=self.alpha_var,
            bg="#1a1a2e", fg="#e0e0e0", troughcolor="#0f3460",
            highlightthickness=0, length=200,
            font=("Consolas", 8)
        )
        self.alpha_scale.pack(side=tk.LEFT)

        self.alpha_info = tk.Label(
            alpha_frame, text="(%70 AI, %30 Kural)",
            font=("Segoe UI", 9), bg="#1a1a2e", fg="#00d4aa"
        )
        self.alpha_info.pack(side=tk.LEFT, padx=8)
        self.alpha_var.trace_add("write", self.update_alpha_label)

        # Durum cubugu
        status_frame = tk.Frame(main, bg="#0f3460")
        status_frame.pack(fill=tk.X, pady=6)

        self.status_label = tk.Label(
            status_frame, text="Hazir",
            font=("Segoe UI", 9), bg="#0f3460", fg="#e0e0e0",
            anchor=tk.W, padx=10, pady=5
        )
        self.status_label.pack(fill=tk.X)

        # Metrikler
        metrics_frame = tk.Frame(main, bg="#16213e", padx=12, pady=8)
        metrics_frame.pack(fill=tk.X, pady=4)

        self.metrics_label = tk.Label(
            metrics_frame,
            text="Model: Egitilmedi | v1 Skor: -- | v2 Skor: -- | Iyilesme: --",
            font=("Consolas", 10, "bold"), bg="#16213e", fg="#00d4aa",
            anchor=tk.W
        )
        self.metrics_label.pack(fill=tk.X)

        # Notebook (sekmeler)
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=8)

        # Sekme 1: Karsilastirma tablosu
        compare_frame = tk.Frame(self.notebook, bg="#1a1a2e")
        self.notebook.add(compare_frame, text="  v1 vs v2 Karsilastirma  ")

        tree_frame1 = tk.Frame(compare_frame, bg="#1a1a2e")
        tree_frame1.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        sb1_y = ttk.Scrollbar(tree_frame1, orient=tk.VERTICAL)
        sb1_x = ttk.Scrollbar(tree_frame1, orient=tk.HORIZONTAL)

        self.compare_tree = ttk.Treeview(
            tree_frame1, yscrollcommand=sb1_y.set, xscrollcommand=sb1_x.set,
            columns=("Ogrenci", "v1_Oda", "v1_Skor", "v2_Oda", "v2_Skor", "Fark", "Durum"),
            show="headings", height=14, style="Custom.Treeview"
        )
        sb1_y.config(command=self.compare_tree.yview)
        sb1_x.config(command=self.compare_tree.xview)

        headings = [
            ("Ogrenci", 160), ("v1_Oda", 90), ("v1_Skor", 80),
            ("v2_Oda", 90), ("v2_Skor", 80), ("Fark", 80), ("Durum", 80)
        ]
        for col, w in headings:
            self.compare_tree.heading(col, text=col.replace("_", " "))
            self.compare_tree.column(col, width=w, anchor=tk.CENTER)

        self.compare_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb1_y.pack(side=tk.RIGHT, fill=tk.Y)
        sb1_x.pack(side=tk.BOTTOM, fill=tk.X)

        # Sekme 2: Feature Importance
        fi_frame = tk.Frame(self.notebook, bg="#1a1a2e")
        self.notebook.add(fi_frame, text="  Ozellik Onem Siralamasi  ")

        fi_inner = tk.Frame(fi_frame, bg="#1a1a2e")
        fi_inner.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        sb2_y = ttk.Scrollbar(fi_inner, orient=tk.VERTICAL)

        self.fi_tree = ttk.Treeview(
            fi_inner, yscrollcommand=sb2_y.set,
            columns=("Ozellik", "Onem", "Gorsel"),
            show="headings", height=14, style="Custom.Treeview"
        )
        sb2_y.config(command=self.fi_tree.yview)

        self.fi_tree.heading("Ozellik", text="Ozellik")
        self.fi_tree.heading("Onem", text="Onem Degeri")
        self.fi_tree.heading("Gorsel", text="Gorsel")
        self.fi_tree.column("Ozellik", width=280, anchor=tk.W)
        self.fi_tree.column("Onem", width=100, anchor=tk.CENTER)
        self.fi_tree.column("Gorsel", width=300, anchor=tk.W)

        self.fi_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb2_y.pack(side=tk.RIGHT, fill=tk.Y)

        # Sekme 3: Model log
        log_frame = tk.Frame(self.notebook, bg="#1a1a2e")
        self.notebook.add(log_frame, text="  Islem Logu  ")

        self.log_text = tk.Text(
            log_frame, bg="#0d1117", fg="#c9d1d9",
            font=("Consolas", 9), wrap=tk.WORD,
            insertbackground="#e0e0e0", relief=tk.FLAT,
            padx=10, pady=8
        )
        log_sb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_sb.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y, pady=4)

    # ---- Yardimci Metodlar ----

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def update_status(self, text):
        self.status_label.config(text=text)
        self.root.update()

    def update_alpha_label(self, *args):
        a = self.alpha_var.get()
        self.alpha_info.config(text=f"(%{a*100:.0f} AI, %{(1-a)*100:.0f} Kural)")

    def _pick_csv(self, title):
        return filedialog.askopenfilename(
            title=title,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

    # ---- Veri Yukleme ----

    def load_rooms(self):
        try:
            path = self._pick_csv("rooms.csv dosyasini sec")
            if not path:
                return
            self.rooms_df = pd.read_csv(path)
            C = Columns()
            if C.current_occupancy not in self.rooms_df.columns:
                self.rooms_df[C.current_occupancy] = 0
            n = len(self.rooms_df)
            self.update_status(f"Oda verileri yuklendi: {n} oda")
            self.log(f"[YUKLE] {n} oda yuklendi: {os.path.basename(path)}")
            messagebox.showinfo("Basarili", f"Oda verileri yuklendi: {n} oda")
        except Exception as e:
            messagebox.showerror("Hata", f"Oda verisi yuklenemedi: {e}")

    def load_students(self):
        try:
            path = self._pick_csv("new_students.csv dosyasini sec")
            if not path:
                return
            self.students_df = pd.read_csv(path)
            n = len(self.students_df)
            self.update_status(f"Ogrenci verileri yuklendi: {n} ogrenci")
            self.log(f"[YUKLE] {n} ogrenci yuklendi: {os.path.basename(path)}")
            messagebox.showinfo("Basarili", f"Ogrenci verileri yuklendi: {n} ogrenci")
        except Exception as e:
            messagebox.showerror("Hata", f"Ogrenci verisi yuklenemedi: {e}")

    def handle_feedback(self):
        choice = messagebox.askyesnocancel(
            "Feedback Verisi",
            "Sentetik geri bildirim verisi uretilsin mi?\n\n"
            "EVET: 5 yillik sentetik veri uret\n"
            "HAYIR: Mevcut feedback CSV yule\n"
            "IPTAL: Vazgec"
        )
        if choice is None:
            return
        elif choice:
            self.generate_synthetic_feedback()
        else:
            self.load_existing_feedback()

    def generate_synthetic_feedback(self):
        if self.rooms_df is None:
            messagebox.showerror("Hata", "Once oda verilerini yukleyin.")
            return

        try:
            self.update_status("Sentetik veri uretiliyor...")
            self.log("[VERI] 5 yillik sentetik geri bildirim uretiliyor...")

            # Oda verisini gecici kaydet
            temp_rooms = os.path.join(BASE_DIR, "_temp_rooms.csv")
            self.rooms_df.to_csv(temp_rooms, index=False)

            self.feedback_df = generate_feedback(
                rooms_path="_temp_rooms.csv",
                n_students_per_year=50,
                n_years=5,
                output_path="feedback_5years.csv",
                seed=42,
            )

            if os.path.exists(temp_rooms):
                os.remove(temp_rooms)

            n = len(self.feedback_df)
            avg = self.feedback_df["satisfaction"].mean()
            self.log(f"[VERI] {n} kayit uretildi, ort. memnuniyet: {avg:.4f}")
            self.update_status(f"Sentetik veri uretildi: {n} kayit")
            messagebox.showinfo("Basarili", f"{n} kayit uretildi.")
        except Exception as e:
            messagebox.showerror("Hata", f"Veri uretilemedi: {e}")
            self.log(f"[HATA] {e}")

    def load_existing_feedback(self):
        try:
            path = self._pick_csv("Feedback CSV dosyasini sec")
            if not path:
                return
            self.feedback_df = pd.read_csv(path)
            n = len(self.feedback_df)
            self.log(f"[YUKLE] Feedback yuklendi: {n} kayit ({os.path.basename(path)})")
            self.update_status(f"Feedback yuklendi: {n} kayit")
            messagebox.showinfo("Basarili", f"Feedback yuklendi: {n} kayit")
        except Exception as e:
            messagebox.showerror("Hata", f"Feedback yuklenemedi: {e}")

    # ---- Model Egitimi ----

    def train_ai_model(self):
        if self.feedback_df is None:
            fb_path = os.path.join(BASE_DIR, "feedback_5years.csv")
            if os.path.exists(fb_path):
                self.feedback_df = pd.read_csv(fb_path)
                self.log("[YUKLE] Mevcut feedback_5years.csv yuklendi")
            else:
                messagebox.showerror("Hata", "Once feedback verisi uretin veya yukleyin.")
                return

        try:
            self.update_status("Model egitiliyor...")
            self.log("[EGITIM] Random Forest modeli egitiliyor...")

            self.model, metrics = train_model(
                feedback_path="feedback_5years.csv",
                n_estimators=200,
                cv_folds=5,
            )

            self.log(f"[EGITIM] R2 (egitim): {metrics['r2_train']:.4f}")
            self.log(f"[EGITIM] RMSE (egitim): {metrics['rmse_train']:.4f}")
            self.log(f"[EGITIM] R2 (CV 5-fold): {metrics['cv_r2_mean']:.4f} +/- {metrics['cv_r2_std']:.4f}")

            save_model(self.model)
            self.log("[EGITIM] Model kaydedildi: trained_model.joblib")

            # Feature importance tablosunu guncelle
            fi = get_feature_importance(self.model)
            for item in self.fi_tree.get_children():
                self.fi_tree.delete(item)
            for _, row in fi.iterrows():
                bar = "#" * int(row["importance"] * 40)
                self.fi_tree.insert("", tk.END, values=(
                    row["feature"], f"{row['importance']:.4f}", bar
                ))

            self.update_status("Model egitildi ve kaydedildi")
            self.notebook.select(1)  # Feature importance sekmesine gec
            messagebox.showinfo("Basarili",
                f"Model egitildi!\n"
                f"R2 (egitim): {metrics['r2_train']:.4f}\n"
                f"R2 (CV): {metrics['cv_r2_mean']:.4f}")
        except Exception as e:
            messagebox.showerror("Hata", f"Egitim hatasi: {e}")
            self.log(f"[HATA] {e}")

    # ---- Eslestirme ----

    def run_v2_matching(self):
        if self.rooms_df is None:
            messagebox.showerror("Hata", "Once oda verilerini yukleyin.")
            return
        if self.students_df is None:
            messagebox.showerror("Hata", "Once ogrenci verilerini yukleyin.")
            return

        # Model yukleme
        if self.model is None:
            model_path = os.path.join(BASE_DIR, "trained_model.joblib")
            if os.path.exists(model_path):
                self.model = load_model()
                self.log("[YUKLE] Mevcut model yuklendi: trained_model.joblib")
            else:
                messagebox.showerror("Hata", "Once modeli egitin.")
                return

        try:
            self.update_status("Eslestirme yapiliyor...")
            self.log(f"[ESLESTIR] v1 ve v2 eslestirme basliyor (alpha={self.alpha_var.get():.2f})...")

            C = Columns()
            W = Weights()
            H = Hard()
            T = TimeCfg()
            S = ScoringCfg()

            rooms = self.rooms_df.copy()
            students = self.students_df.copy()

            # v1 eslestirme
            self.results_v1, flow_v1, score_v1 = batch_assign(
                students, rooms, C, W, H, T, S
            )
            self.log(f"[v1] {flow_v1} ogrenci atandi, toplam skor: {score_v1:.3f}")

            # v2 eslestirme
            alpha = self.alpha_var.get()
            self.results_v2, flow_v2, score_v2 = batch_assign_v2(
                students, rooms, self.model, C, W, H, T, S,
                alpha=alpha, use_hybrid=True
            )
            self.log(f"[v2] {flow_v2} ogrenci atandi, toplam skor: {score_v2:.3f}")

            # v2 sonuclarini kaydet
            self.results_v2.to_csv(os.path.join(BASE_DIR, "assignments_v2.csv"), index=False)

            # Metrikleri guncelle
            v1_avg = self.results_v1[self.results_v1["status"] == "assigned"]["score"].mean()
            v2_avg = self.results_v2[self.results_v2["status"] == "assigned"]["score"].mean()
            improvement = ((v2_avg - v1_avg) / v1_avg) * 100 if v1_avg > 0 else 0

            self.metrics_label.config(
                text=f"Model: Aktif | v1 Ort: {v1_avg:.4f} | v2 Ort: {v2_avg:.4f} | "
                     f"Iyilesme: {improvement:+.1f}%"
            )

            # Karsilastirma tablosunu doldur
            for item in self.compare_tree.get_children():
                self.compare_tree.delete(item)

            merged = self.results_v1.merge(
                self.results_v2, on=C.student_id, suffixes=("_v1", "_v2")
            )

            for _, row in merged.iterrows():
                name = row[C.student_id]
                r1 = row.get("assigned_room_id_v1", "N/A")
                s1 = float(row.get("score_v1", 0)) if pd.notna(row.get("score_v1")) else 0
                r2 = row.get("assigned_room_id_v2", "N/A")
                s2 = float(row.get("score_v2", 0)) if pd.notna(row.get("score_v2")) else 0
                diff = s2 - s1

                if diff > 0.001:
                    indicator = "Artti"
                elif diff < -0.001:
                    indicator = "Azaldi"
                else:
                    indicator = "Ayni"

                self.compare_tree.insert("", tk.END, values=(
                    name,
                    str(r1) if pd.notna(r1) else "N/A",
                    f"{s1:.4f}",
                    str(r2) if pd.notna(r2) else "N/A",
                    f"{s2:.4f}",
                    f"{diff:+.4f}",
                    indicator,
                ))

            v1_count = (self.results_v1["status"] == "assigned").sum()
            v2_count = (self.results_v2["status"] == "assigned").sum()
            self.update_status(
                f"Eslestirme tamamlandi | v1: {v1_count} atandi | v2: {v2_count} atandi | "
                f"Iyilesme: {improvement:+.1f}%"
            )
            self.notebook.select(0)  # Karsilastirma sekmesine gec

            self.log(f"[SONUC] v1 ort={v1_avg:.4f}, v2 ort={v2_avg:.4f}, iyilesme={improvement:+.1f}%")
            messagebox.showinfo("Basarili",
                f"Eslestirme tamamlandi!\n\n"
                f"v1 Ortalama: {v1_avg:.4f}\n"
                f"v2 Ortalama: {v2_avg:.4f}\n"
                f"Iyilesme: {improvement:+.1f}%")
        except Exception as e:
            messagebox.showerror("Hata", f"Eslestirme hatasi: {e}")
            self.log(f"[HATA] {e}")
            import traceback
            self.log(traceback.format_exc())

    def save_results(self):
        if self.results_v2 is None:
            messagebox.showerror("Hata", "Kaydedilecek sonuc yok. Once eslestirme yapin.")
            return

        try:
            path = filedialog.asksaveasfilename(
                title="Sonuc dosyasini kaydet",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not path:
                return
            self.results_v2.to_csv(path, index=False)
            self.update_status(f"Sonuclar kaydedildi: {os.path.basename(path)}")
            self.log(f"[KAYDET] Sonuclar kaydedildi: {path}")
            messagebox.showinfo("Basarili", f"Sonuclar kaydedildi:\n{path}")
        except Exception as e:
            messagebox.showerror("Hata", f"Kaydetme hatasi: {e}")


def main():
    root = tk.Tk()
    app = KykMatcherV2App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
