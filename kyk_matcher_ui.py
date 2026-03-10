import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

from kyk_matcher_v1 import Columns, Weights, Hard, TimeCfg, ScoringCfg, batch_assign


class KykMatcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KYK Oda Eşleştirme (Karar Algoritması v1)")
        self.root.geometry("1100x720")
        self.root.configure(bg="#f0f0f0")

        self.rooms_df = None
        self.students_df = None
        self.results_df = None

        self.flow = 0
        self.total_score = 0

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        main_frame = tk.Frame(self.root, bg="#f0f0f0", padx=18, pady=18)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(
            main_frame,
            text="KYK Oda Eşleştirme Sistemi (v1)",
            font=("Arial", 18, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50"
        )
        title.pack(pady=(0, 14))

        button_frame = tk.Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(fill=tk.X, pady=8)

        tk.Button(
            button_frame,
            text="Oda CSV Yükle",
            command=self.load_rooms,
            bg="#2ecc71",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=14,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            button_frame,
            text="Yeni Öğrenci CSV Yükle",
            command=self.load_students,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=14,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            button_frame,
            text="Eşleştirme Çalıştır",
            command=self.run_matching,
            bg="#f39c12",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=14,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            button_frame,
            text="Sonuçları Kaydet",
            command=self.save_results,
            bg="#16a085",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=14,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=6)

        status_frame = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill=tk.X, pady=10)

        self.status_label = tk.Label(
            status_frame,
            text="Hazır",
            font=("Arial", 9),
            bg="#ecf0f1",
            anchor=tk.W,
            padx=10,
            pady=6
        )
        self.status_label.pack(fill=tk.X)

        metrics_frame = tk.Frame(main_frame, bg="#f0f0f0")
        metrics_frame.pack(fill=tk.X, pady=(0, 8))

        self.metrics_label = tk.Label(
            metrics_frame,
            text="Flow: 0 | Toplam Uyum Skoru: 0",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
            anchor=tk.W
        )
        self.metrics_label.pack(fill=tk.X)

        results_frame = tk.LabelFrame(
            main_frame,
            text="Eşleştirme Sonuçları",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
            padx=10,
            pady=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        tree_frame = tk.Frame(results_frame, bg="#f0f0f0")
        tree_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)

        self.tree = ttk.Treeview(
            tree_frame,
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set,
            columns=("Öğrenci", "Oda", "Skor", "Durum"),
            show="headings",
            height=16
        )

        scrollbar_y.config(command=self.tree.yview)
        scrollbar_x.config(command=self.tree.xview)

        self.tree.heading("Öğrenci", text="Öğrenci")
        self.tree.heading("Oda", text="Atanan Oda")
        self.tree.heading("Skor", text="Uyum Skoru")
        self.tree.heading("Durum", text="Durum")

        self.tree.column("Öğrenci", width=240, anchor=tk.CENTER)
        self.tree.column("Oda", width=140, anchor=tk.CENTER)
        self.tree.column("Skor", width=120, anchor=tk.CENTER)
        self.tree.column("Durum", width=220, anchor=tk.CENTER)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status(self, text):
        self.status_label.config(text=text)
        self.root.update()

    def _pick_csv(self, title):
        return filedialog.askopenfilename(
            title=title,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

    def load_rooms(self):
        try:
            path = self._pick_csv("rooms.csv dosyasını seç")
            if not path:
                return
            df = pd.read_csv(path)
            self.rooms_df = df
            self.update_status(f"Oda verileri yüklendi: {len(df)} oda")
            messagebox.showinfo("Başarılı", f"Oda verileri yüklendi: {len(df)} oda")
        except Exception as e:
            messagebox.showerror("Hata", f"Oda verisi yüklenemedi: {str(e)}")
            self.update_status("Hata: Oda verisi yüklenemedi")

    def load_students(self):
        try:
            path = self._pick_csv("new_students.csv dosyasını seç")
            if not path:
                return
            df = pd.read_csv(path)
            self.students_df = df
            self.update_status(f"Yeni öğrenci verileri yüklendi: {len(df)} öğrenci")
            messagebox.showinfo("Başarılı", f"Yeni öğrenci verileri yüklendi: {len(df)} öğrenci")
        except Exception as e:
            messagebox.showerror("Hata", f"Öğrenci verisi yüklenemedi: {str(e)}")
            self.update_status("Hata: Öğrenci verisi yüklenemedi")

    def run_matching(self):
        if self.rooms_df is None:
            messagebox.showerror("Hata", "Önce oda verilerini yükleyin.")
            return
        if self.students_df is None:
            messagebox.showerror("Hata", "Önce yeni öğrenci verilerini yükleyin.")
            return

        try:
            for item in self.tree.get_children():
                self.tree.delete(item)

            self.update_status("Eşleştirme çalışıyor...")

            C = Columns()
            W = Weights()
            H = Hard()
            T = TimeCfg()
            S = ScoringCfg()

            rooms = self.rooms_df.copy()
            students = self.students_df.copy()

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

            out, flow, total_score = batch_assign(students, rooms, C, W, H, T, S)

            self.results_df = out
            self.flow = int(flow)
            self.total_score = float(total_score)

            self.metrics_label.config(text=f"Flow: {self.flow} | Toplam Uyum Skoru: {self.total_score:.3f}")

            for _, row in out.iterrows():
                sid = row[C.student_id] if C.student_id in out.columns else ""
                rid = row["assigned_room_id"]
                score = row["score"]
                status = row["status"]
                score_disp = "N/A" if pd.isna(score) else f"{float(score):.3f}"
                rid_disp = "N/A" if pd.isna(rid) else str(rid)
                self.tree.insert("", tk.END, values=(str(sid), rid_disp, score_disp, str(status)))

            assigned_count = int((out["status"] == "assigned").sum())
            self.update_status(f"Eşleştirme tamamlandı: {assigned_count}/{len(out)} atandı")
            messagebox.showinfo("Başarılı", f"Eşleştirme tamamlandı.\nAtanan: {assigned_count}/{len(out)}\nFlow: {self.flow}\nToplam skor: {self.total_score:.3f}")
        except Exception as e:
            messagebox.showerror("Hata", f"Eşleştirme hatası: {str(e)}")
            self.update_status("Hata: Eşleştirme yapılamadı")

    def save_results(self):
        if self.results_df is None:
            messagebox.showerror("Hata", "Kaydedilecek sonuç yok. Önce eşleştirme çalıştırın.")
            return

        try:
            path = filedialog.asksaveasfilename(
                title="Sonuç dosyasını kaydet",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not path:
                return
            self.results_df.to_csv(path, index=False)
            self.update_status(f"Sonuçlar kaydedildi: {os.path.basename(path)}")
            messagebox.showinfo("Başarılı", f"Sonuçlar kaydedildi:\n{path}")
        except Exception as e:
            messagebox.showerror("Hata", f"Kaydetme hatası: {str(e)}")
            self.update_status("Hata: Kaydedilemedi")


def main():
    root = tk.Tk()
    app = KykMatcherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
