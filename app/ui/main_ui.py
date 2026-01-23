import tkinter as tk
import threading
from multiprocessing import Process

import app.ui.pointRecorder as pointRecorder
import app.ui.macroRecorder as macroRecorder
import app.ui.train as train
import app.ui.inferece as inference

from app.ui.plot import plot_main
from app.repostitories.DBController import point_clear, read, macro_point_clear


class MouseMacroUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Mouse Macro Tool")
        self.geometry("500x750")
        self.configure(bg="#2E3440")
        self.resizable(False, False)

        self.stop_recording = threading.Event()
        self.stop_train = threading.Event()
        self.stop_inference_event = threading.Event()

        self.init_ui()

    # ================= UI Helpers =================
    def create_section(self, parent, title):
        frame = tk.Frame(parent, bg="#3B4252", padx=10, pady=10)
        frame.pack(fill="x", pady=10)

        label = tk.Label(
            frame,
            text=title,
            font=("Helvetica", 14, "bold"),
            bg="#3B4252",
            fg="#ECEFF4"
        )
        label.pack(anchor="w", pady=(0, 10))

        btn_area = tk.Frame(frame, bg="#3B4252")
        btn_area.pack()

        return btn_area

    def create_button(self, parent, text, cmd, row, col, bg):
        btn = tk.Label(
            parent,
            text=text,
            bg=bg,
            fg="#2E3440",
            font=("Helvetica", 12, "bold"),
            width=16,
            height=2,
            relief="raised",
            bd=0
        )

        btn.grid(row=row, column=col, padx=6, pady=6)

        btn.bind("<Button-1>", lambda e: cmd())
        btn.bind("<Enter>", lambda e: btn.config(bg="#81A1C1"))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))

    # ================= UI Layout =================
    def init_ui(self):
        title_label = tk.Label(
            self,
            text="🐭 Mouse Macro Tool",
            font=("Helvetica", 20, "bold"),
            bg="#2E3440",
            fg="#D8DEE9"
        )
        title_label.pack(pady=(20, 10))

        card_frame = tk.Frame(self, bg="#434C5E", padx=15, pady=15)
        card_frame.pack(padx=20, pady=10, fill="both", expand=True)

        # 🎥 Recording
        record_area = self.create_section(card_frame, "🎥 Recording")
        self.create_button(record_area, "Mouse Record", self.start_record, 0, 0, "#88C0D0")
        self.create_button(record_area, "Macro Record", self.start_macro_record, 0, 1, "#88C0D0")
        self.create_button(record_area, "Stop Record", self.stop_record, 1, 0, "#EBCB8B")

        # 📊 Plot
        plot_area = self.create_section(card_frame, "📊 Plot")
        self.create_button(plot_area, "Mouse Plot", lambda: self.make_plot_in_process(True), 0, 0, "#A3BE8C")
        self.create_button(plot_area, "Macro Plot", lambda: self.make_plot_in_process(False), 0, 1, "#A3BE8C")

        # 🧠 AI
        ai_area = self.create_section(card_frame, "🧠 AI")
        self.create_button(ai_area, "Train", self.start_train, 0, 0, "#5E81AC")
        self.create_button(ai_area, "Stop Train", self.stop_training, 0, 1, "#BF616A")
        self.create_button(ai_area, "Inference", self.start_inference, 1, 0, "#81A1C1")
        self.create_button(ai_area, "Stop Inference", self.stop_inference, 1, 1, "#BF616A")

        # 🧹 Database
        db_area = self.create_section(card_frame, "🧹 Database")
        self.create_button(db_area, "Mouse Clear", self.clear_db, 0, 0, "#D08770")
        self.create_button(db_area, "Macro Clear", self.macro_clear_db, 0, 1, "#D08770")

        footer = tk.Label(
            self,
            text="v1.1 - Mouse Macro Tool",
            font=("Helvetica", 10),
            bg="#2E3440",
            fg="#D8DEE9"
        )
        footer.pack(side="bottom", pady=10)

    # ================= Logic =================
    def make_plot_in_process(self, user=False):
        points = read(user)
        p = Process(target=plot_main, args=(points,))
        p.start()

    def start_macro_record(self):
        self.stop_recording.clear()
        threading.Thread(
            target=macroRecorder.main,
            args=(self.stop_recording,),
            daemon=True
        ).start()

    def start_record(self):
        self.stop_recording.clear()
        threading.Thread(
            target=pointRecorder.main,
            args=(self.stop_recording,),
            daemon=True
        ).start()

    def start_train(self):
        self.stop_train.clear()
        threading.Thread(
            target=train.main,
            args=(self.stop_train,),
            daemon=True
        ).start()

    def start_inference(self):
        self.stop_inference_event.clear()
        threading.Thread(
            target=inference.main,
            args=(self.stop_inference_event,),
            daemon=True
        ).start()

    def stop_training(self):
        self.stop_train.set()
        print("학습 중지 요청됨")

    def stop_inference(self):
        self.stop_inference_event.set()
        print("매크로 탐지 중지 요청됨")

    def stop_record(self):
        self.stop_recording.set()
        print("마우스 기록 중지 요청됨")

    def clear_db(self):
        point_clear()
        print("Mouse DB 초기화")

    def macro_clear_db(self):
        macro_point_clear()
        print("Macro DB 초기화")
