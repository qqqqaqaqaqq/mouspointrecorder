import os

import tkinter as tk
import threading
from multiprocessing import Process
from tkinter import messagebox

import app.ui.train as train
import app.ui.inferece as inference
import app.core.globals as globals

from app.ui.plot import plot_main
from app.repostitories.DBController import point_clear, read, macro_point_clear
from app.services.macroMouse import record_mouse_path

import app.services.userMouse as useMouse

from multiprocessing import Process, Event

class MouseMacroUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Mouse Macro Tool")
        self.geometry("1440x980")
        self.minsize(1440, 980)
        self.configure(bg="#2E3440")
        self.resizable(True, True)

        self.stop_train = threading.Event()
        self.stop_inference_event = threading.Event()
        self.stop_move_event = Event()

        self.init_ui()

    # ================= UI Helpers =================
    def create_section(self, parent, title, bg="#3B4252", fg="#ECEFF4"):
        frame = tk.Frame(parent, bg=bg, padx=10, pady=10)
        frame.pack(fill="x", pady=10)

        label = tk.Label(
            frame,
            text=title,
            font=("Helvetica", 14, "bold"),
            bg=bg,
            fg=fg
        )
        label.pack(anchor="w", pady=(0, 10))

        btn_area = tk.Frame(frame, bg=bg)
        btn_area.pack()

        return btn_area

    def create_button(self, parent, text, cmd, row, col, bg):
        btn = tk.Label(
            parent,
            text=text,
            bg=bg,
            fg="#2E3440",
            font=("Helvetica", 10, "bold"),
            width=40,
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
        # 🐭 Title
        title_label = tk.Label(
            self,
            text="🐭 Mouse Macro Tool",
            font=("Helvetica", 22, "bold"),
            bg="#2E3440",
            fg="#ECEFF4"
        )
        title_label.pack(pady=(20, 15))

        # Main Frame
        main_frame = tk.Frame(self, bg="#434C5E")
        main_frame.pack(padx=20, pady=10, fill="both", expand=True)

        # ===================== LEFT PANEL (기능) =====================
        left_frame = tk.Frame(main_frame, bg="#434C5E")
        left_frame.pack(side="left", fill="both", expand=True)

        # --- Recording Section ---
        record_area = self.create_section(left_frame, "🎥 Recording : Exit Key : q", bg="#4C566A", fg="#ECEFF4")
        buttons_info = [
            ("Mouse Record", lambda: self.start_record(record=True)),
            ("Macro Record Move False", lambda: self.start_macro_record_move_false(move=False, record=True)),
            ("Macro Record Move True", lambda: self.start_macro_record_move_true(move=True, record=True)),
            ("Macro Move", lambda: self.start_macro_move(move=True, record=False)),
        ]
        colors = ["#88C0D0", "#88C0D0", "#88C0D0", "#88C0D0"]
        for idx, (text, cmd) in enumerate(buttons_info):
            row, col = divmod(idx, 2)
            self.create_button(record_area, text, cmd, row, col, colors[idx])

        # --- SEQ_LEN / STRIDE Section ---
        seq_frame = tk.Frame(left_frame, bg="#3B4252", padx=15, pady=15, relief="raised", bd=2)
        seq_frame.pack(fill="x", pady=15)

        tk.Label(seq_frame, text="SEQ_LEN:", bg="#3B4252", fg="#ECEFF4", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.seq_entry = tk.Entry(seq_frame, width=6, font=("Helvetica", 12))
        self.seq_entry.grid(row=0, column=1, padx=(5,20))
        self.seq_entry.insert(0, str(globals.SEQ_LEN))

        tk.Label(seq_frame, text="STRIDE:", bg="#3B4252", fg="#ECEFF4", font=("Helvetica", 12, "bold")).grid(row=0, column=2, sticky="w")
        self.stride_entry = tk.Entry(seq_frame, width=6, font=("Helvetica", 12))
        self.stride_entry.grid(row=0, column=3, padx=(5,20))
        self.stride_entry.insert(0, str(globals.STRIDE))

        tk.Button(seq_frame, text="적용", command=self.apply_seq_stride, bg="#A3BE8C", fg="#2E3440", font=("Helvetica", 12, "bold")).grid(row=0, column=4)

        # --- Plot Section ---
        plot_area = self.create_section(left_frame, "📊 Plot", bg="#4C566A", fg="#ECEFF4")
        self.create_button(plot_area, "Mouse Plot", lambda: self.make_plot_in_process(True), 0, 0, "#A3BE8C")
        self.create_button(plot_area, "Macro Plot", lambda: self.make_plot_in_process(False), 0, 1, "#A3BE8C")

        # --- AI Section ---
        ai_area = self.create_section(left_frame, "🧠 AI", bg="#4C566A", fg="#ECEFF4")
        button_frame = tk.Frame(ai_area, bg="#4C566A")
        button_frame.grid(row=0, column=0, sticky="nw")
        self.create_button(button_frame, "Train", self.start_train, 0, 0, "#5E81AC")
        self.create_button(button_frame, "Stop Train", self.stop_training, 0, 1, "#BF616A")
        self.create_button(button_frame, "Inference", self.start_inference, 1, 0, "#81A1C1")
        self.create_button(button_frame, "Stop Inference", self.stop_inference, 1, 1, "#BF616A")

        # --- Database Section ---
        db_area = self.create_section(left_frame, "🧹 Database", bg="#4C566A", fg="#ECEFF4")
        self.create_button(db_area, "Mouse Clear", self.clear_db, 0, 0, "#D08770")
        self.create_button(db_area, "Macro Clear", self.macro_clear_db, 0, 1, "#D08770")

        # ===================== RIGHT PANEL (로그) =====================
        right_frame = tk.Frame(main_frame, bg="#2E3440", width=400)
        right_frame.pack(side="right", fill="y")

        scrollbar = tk.Scrollbar(right_frame)
        scrollbar.pack(side="right", fill="y")

        self.macro_text = tk.Text(
            right_frame,
            width=100,
            bg="#2E3440",
            fg="#ECEFF4",
            font=("Helvetica", 14),
            yscrollcommand=scrollbar.set
        )
        self.macro_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.macro_text.yview)

        # ===== MACRO_DETECTOR 업데이트 시작 =====
        self.update_macro_detector()

        # ===================== Footer =====================
        footer = tk.Label(
            self,
            text="v1.1 - Mouse Macro Tool, Created by qqqa",
            font=("Helvetica", 10, "italic"),
            bg="#2E3440",
            fg="#D8DEE9"
        )
        footer.pack(side="bottom", pady=10)

    # ===================== MACRO_DETECTOR 업데이트 =====================
    def update_macro_detector(self):
        self.macro_text.delete("1.0", tk.END)
        for line in globals.MACRO_DETECTOR:
            self.macro_text.insert(tk.END, f"{line}\n")
        self.macro_text.see(tk.END)
        self.after(200, self.update_macro_detector)
        
    def apply_seq_stride(self):
        try:
            seq_val = int(self.seq_entry.get())
            stride_val = int(self.stride_entry.get())
            if seq_val < 1 or stride_val < 1:
                raise ValueError

            # globals 값 변경
            globals.SEQ_LEN = seq_val
            globals.STRIDE = stride_val
            print(f"[INFO] globals.SEQ_LEN = {globals.SEQ_LEN}, globals.STRIDE = {globals.STRIDE}")

            # env 파일 경로 (프로젝트 root 기준)
            env_path = ".env"

            # 기존 env 읽기
            env_dict = {}
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    for line in f:
                        if "=" in line:
                            key, val = line.strip().split("=", 1)
                            env_dict[key] = val

            # 값 변경
            env_dict["SEQ_LEN"] = str(seq_val)
            env_dict["STRIDE"] = str(stride_val)

            # 다시 쓰기
            with open(env_path, "w") as f:
                for key, val in env_dict.items():
                    f.write(f"{key}={val}\n")

            print(f"[INFO] .env 파일 업데이트 완료: SEQ_LEN={seq_val}, STRIDE={stride_val}")

        except ValueError:
            print("[ERROR] 올바른 정수를 입력하세요")

    # ================= Logic =================
    def make_plot_in_process(self, user=False):
        points = read(user)
        p = Process(target=plot_main, args=(points,))
        p.start()

    def start_macro_record_move_false(self, move=False, record=True):
        result = messagebox.askyesno("확인", "매크로 기록 무브=False 시작하시겠습니까?")
        if result:
            self.stop_move_event.clear()
            p = Process(target=record_mouse_path, kwargs={"move_mouse": move, "record": record, "stop_event": self.stop_move_event})
            p.start()

    def start_macro_record_move_true(self, move=True, record=True):
        result = messagebox.askyesno("확인", "매크로 기록 무브=True 시작하시겠습니까?")
        if result:
            self.stop_move_event.clear()            
            p = Process(target=record_mouse_path, kwargs={"move_mouse": move, "record": record, "stop_event": self.stop_move_event})
            p.start()

    def start_macro_move(self, move=True, record=False):
        result = messagebox.askyesno("확인", "매크로 마우스 기록=False 무브=True 시작하시겠습니까?")
        if result:
            self.stop_move_event.clear()            
            p = Process(target=record_mouse_path, kwargs={"move_mouse": move, "record": record, "stop_event": self.stop_move_event})
            p.start()

    def start_record(self, record=False):
        result = messagebox.askyesno("확인", "유저 마우스 기록=True 시작하시겠습니까?")
        if result:
            self.stop_move_event.clear()            
            p = Process(target=useMouse.record_mouse_path, kwargs={"record": record, "stop_event": self.stop_move_event})
            p.start()

    def start_train(self):
        result = messagebox.askyesno("확인", "학습을 시작하시겠습니까?")
        if result:                           
            self.stop_train.clear()
            threading.Thread(
                target=train.main,
                args=(self.stop_train,),
                daemon=True
            ).start()

    def start_inference(self):
        result = messagebox.askyesno("확인", "탐지를 시작하시겠습니까?")
        if result:            
            self.stop_inference_event.clear()
            threading.Thread(
                target=inference.main,
                args=(self.stop_inference_event,),
                daemon=True
            ).start()

    def stop_training(self):
        result = messagebox.askyesno("확인", "학습을 중지하시겠습니까?")
        if result:                  
            self.stop_train.set()
            print("학습 중지 요청됨")

    def stop_inference(self):
        result = messagebox.askyesno("확인", "탐지를 중지하시겠습니까?")
        if result:                
            self.stop_inference_event.set()
            print("매크로 탐지 중지 요청됨")

    def clear_db(self):
        result = messagebox.askyesno("확인", "Mouse DB를 초기화하시겠습니까?")
        if result:
            point_clear()
            print("Mouse DB 초기화")
            messagebox.showinfo("완료", "초기화가 완료되었습니다.")

    def macro_clear_db(self):
        result = messagebox.askyesno("확인", "Macro DB를 초기화하시겠습니까?")
        if result:        
            macro_point_clear()
            print("Macro DB 초기화")
            messagebox.showinfo("완료", "초기화가 완료되었습니다.")