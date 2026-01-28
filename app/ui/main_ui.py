import tkinter as tk
from tkinter import messagebox
import threading
from multiprocessing import Process, Event
import keyboard
import time, os, sys

import app.ui.train as train
import app.ui.inferece as inference
from app.ui.plot import plot_main

import app.core.globals as globals


import app.repostitories.DBController as DBController
import app.repostitories.JsonController as JsonController
from app.services.macroMouse import record_mouse_path
import app.services.userMouse as useMouse
from app.core.logger import add_macro_log

def restart_program():
    response = messagebox.askyesno("재시작 알림", "프로그램을 재시작합니다.")
    if not response:
        return
    python = sys.executable
    os.execl(python, python, *sys.argv)

class MouseMacroUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mouse Macro Tool")
        self.geometry("1440x980")
        self.minsize(1440, 980)
        self.configure(bg="#1F2024")  # 다크 배경
        self.stop_train = threading.Event()
        self.stop_inference_event = threading.Event()
        self.stop_move_event = Event()
        keyboard.add_hotkey('ctrl+shift+q', lambda: self.stop_move_event.set())    
        self.init_ui()
        
        self.after(100, self.process_macro_logs)

    def process_macro_logs(self):
        while not globals.LOG_QUEUE.empty():
            add_macro_log(globals.LOG_QUEUE.get())
        self.after(100, self.process_macro_logs)
        
    # ================= UI Helpers =================
    def create_section(self, parent, title, bg="#2A2B30", fg="#E0E0E0"):
        frame = tk.Frame(parent, bg=bg, padx=15, pady=15, bd=0, relief="flat")
        frame.pack(fill="x", pady=10)
        frame.configure(highlightbackground="#444", highlightthickness=1, bd=0)

        label = tk.Label(frame, text=title, font=("Helvetica", 14, "bold"), bg=bg, fg=fg)
        label.pack(anchor="w", pady=(0,10))

        btn_area = tk.Frame(frame, bg=bg)
        btn_area.pack()
        return btn_area

    def create_button(self, parent, text, cmd, row, col, bg="#3C3F41"):
        btn = tk.Label(parent, text=text, bg=bg, fg="#FFFFFF", font=("Helvetica", 11, "bold"),
                       width=36, height=2, relief="raised", bd=0, cursor="hand2")
        btn.grid(row=row, column=col, padx=6, pady=6)
        btn.bind("<Button-1>", lambda e: cmd())
        btn.bind("<Enter>", lambda e: btn.config(bg="#5C7AEA"))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    # ================= UI Layout =================
    def init_ui(self):
        # 🐭 Title
        title_label = tk.Label(self, text="🐭 Mouse Macro Tool", font=("Helvetica", 24, "bold"),
                               bg="#1F2024", fg="#FFFFFF")
        title_label.pack(pady=(20,15))

        # Main Frame
        main_frame = tk.Frame(self, bg="#1F2024")
        main_frame.pack(padx=20, pady=10, fill="both", expand=True)

        # ===== LEFT PANEL =====
        left_frame = tk.Frame(main_frame, bg="#1F2024")
        left_frame.pack(side="left", fill="both", expand=True)

        # --- Recording Section ---
        record_area = self.create_section(left_frame, "🎥 Recording (Exit Key: Ctrl + Shift + Q)")
        buttons_info = [
            ("Mouse Record", lambda: self.start_record(record=True)),
            ("Macro Record Move False", lambda: self.start_macro_record_move_false(move=False, user_macro=True, record=True)),
            ("User Macro Record", lambda: self.start_macro_record_move_false(move=False, user_macro=True, record=True)),            
            ("Macro Record Move True", lambda: self.start_macro_record_move_true(move=True, record=True)),
            ("Macro Move", lambda: self.start_macro_move(move=True, record=False)),      
        ]
        colors = ["#5C7AEA"]*5
        for idx, (text, cmd) in enumerate(buttons_info):
            row, col = divmod(idx, 2)
            self.create_button(record_area, text, cmd, row, col, colors[idx])

        # --- SEQ_LEN / STRIDE Section ---
        seq_frame = tk.Frame(left_frame, bg="#2A2B30", padx=10, pady=10)
        seq_frame.pack(fill="x", pady=15)

        tk.Label(seq_frame, text="SEQ_LEN:", bg="#2A2B30", fg="#E0E0E0", font=("Helvetica", 12, "bold")).grid(row=0, column=0)
        self.seq_entry = tk.Entry(seq_frame, width=6, font=("Helvetica", 12))
        self.seq_entry.grid(row=0, column=1, padx=(5,20))
        self.seq_entry.insert(0, str(globals.SEQ_LEN))

        tk.Label(seq_frame, text="STRIDE:", bg="#2A2B30", fg="#E0E0E0", font=("Helvetica", 12, "bold")).grid(row=0, column=2)
        self.stride_entry = tk.Entry(seq_frame, width=6, font=("Helvetica", 12))
        self.stride_entry.grid(row=0, column=3, padx=(5,20))
        self.stride_entry.insert(0, str(globals.STRIDE))

        tk.Label(seq_frame, text="threshold:", bg="#2A2B30", fg="#E0E0E0", font=("Helvetica", 12, "bold")).grid(row=0, column=4)
        self.threshold_entry = tk.Entry(seq_frame, width=6, font=("Helvetica", 12))
        self.threshold_entry.grid(row=0, column=5, padx=(5,20))
        self.threshold_entry.insert(0, str(globals.threshold))

        tk.Button(seq_frame, text="적용", command=self.apply_seq_stride, bg="#4CBB17", fg="#FFFFFF", font=("Helvetica", 12, "bold")).grid(row=0, column=6, padx=(5,0))

        self.toggle_btn = tk.Button(seq_frame, text=f"저장: {globals.Recorder}", command=self.toggle_record_path,
                                    bg="#FF6F61", fg="#FFFFFF", font=("Helvetica", 12, "bold"))
        self.toggle_btn.grid(row=0, column=7, padx=(10,0))

        # --- Plot Section ---
        plot_area = self.create_section(left_frame, "📊 Plot")
        self.create_button(plot_area, "Mouse Plot", lambda: self.make_plot_in_process(True), 0, 0, "#4CBB17")
        self.create_button(plot_area, "Macro Plot", lambda: self.make_plot_in_process(False), 0, 1, "#4CBB17")

        # --- AI Section ---
        ai_area = self.create_section(left_frame, "🧠 AI")
        button_frame = tk.Frame(ai_area, bg="#2A2B30")
        button_frame.grid(row=0, column=0, sticky="nw")
        self.create_button(button_frame, "Train", self.start_train, 0, 0, "#5C7AEA")
        self.create_button(button_frame, "Stop Train", self.stop_training, 0, 1, "#E94B3C")
        self.create_button(button_frame, "Inference", self.start_inference, 1, 0, "#81C0F7")
        self.create_button(button_frame, "Stop Inference", self.stop_inference, 1, 1, "#E94B3C")

        # --- Database Section ---
        db_area = self.create_section(left_frame, "🧹 Database")
        self.create_button(db_area, "Mouse Clear", self.clear_db, 0, 0, "#FF8C42")
        self.create_button(db_area, "Macro Clear", self.macro_clear_db, 0, 1, "#FF8C42")

        # ===== RIGHT PANEL (로그) =====
        right_frame = tk.Frame(main_frame, bg="#1F2024", width=400)
        right_frame.pack(side="right", fill="y")

        scrollbar = tk.Scrollbar(right_frame)
        scrollbar.pack(side="right", fill="y")

        self.macro_text = tk.Text(right_frame, width=100, bg="#2A2B30", fg="#E0E0E0",
                                  font=("Helvetica", 14), yscrollcommand=scrollbar.set, relief="flat")
        self.macro_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.macro_text.yview)

        self.update_macro_detector()

        footer = tk.Label(self, text="v1.0.3 - Mouse Macro Tool, Created by qqqa", font=("Helvetica", 10, "italic"),
                          bg="#1F2024", fg="#A0A0A0")
        footer.pack(side="bottom", pady=10)


    # ===================== MACRO_DETECTOR 업데이트 =====================
    def update_macro_detector(self):
        self.macro_text.delete("1.0", tk.END)
        for line in globals.MACRO_DETECTOR:
            self.macro_text.insert(tk.END, f"{line}\n")
        self.macro_text.see(tk.END)
        self.after(200, self.update_macro_detector)
    
    # ===================== MouseMacroUI 클래스 안 =====================

    def toggle_record_path(self):
        # RecordPath 토글
        if getattr(globals, "Recorder", "json") == "json":
            globals.Recorder = "postgres"
        else:
            globals.Recorder = "json"
        
        globals.LOG_QUEUE.put(f"[INFO] 저장 타입 변경: {globals.Recorder}")
        
        # 버튼 텍스트 업데이트
        if hasattr(self, "toggle_btn"):
            self.toggle_btn.config(text=f"저장: {globals.Recorder}")

        # .env 업데이트
        env_path = ".env"
        env_dict = {}
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        key, val = line.strip().split("=", 1)
                        env_dict[key] = val
        
        env_dict["Recorder"] = globals.Recorder

        with open(env_path, "w", encoding="utf-8") as f:
            for key, val in env_dict.items():
                f.write(f"{key}={val}\n")

        globals.LOG_QUEUE.put(f"[INFO] .env 파일 업데이트 완료: Recorder={globals.Recorder}")
        globals.LOG_QUEUE.put(f"3초후 프로그램이 재부팅 됩니다.")
        time.sleep(3)
        restart_program()

    def apply_seq_stride(self):
        try:
            seq_val = int(self.seq_entry.get())
            stride_val = int(self.stride_entry.get())
            threshold_val = float(self.threshold_entry.get())
            if seq_val < 1 or stride_val < 1:
                raise ValueError

            # globals 값 변경
            globals.SEQ_LEN = seq_val
            globals.STRIDE = stride_val
            globals.threshold = threshold_val
            globals.LOG_QUEUE.put(f"[INFO] globals.SEQ_LEN = {globals.SEQ_LEN}, globals.STRIDE = {globals.STRIDE}, globals.threshold = {globals.threshold}")

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
            env_dict["threshold"] = str(threshold_val)

            # 다시 쓰기
            with open(env_path, "w") as f:
                for key, val in env_dict.items():
                    f.write(f"{key}={val}\n")

            globals.LOG_QUEUE.put(f"[INFO] .env 파일 업데이트 완료: SEQ_LEN={seq_val}, STRIDE={stride_val}, threshold={threshold_val}")

        except ValueError:
            globals.LOG_QUEUE.put("[ERROR] 올바른 정수를 입력하세요")

    # ================= Logic =================
    def make_plot_in_process(self, user=False):
        if globals.Recorder == "postgres":
            points = DBController.read(user, log_queue=globals.LOG_QUEUE)
        elif globals.Recorder == 'json':
            points = JsonController.read(user, log_queue=globals.LOG_QUEUE)

        p = Process(
            target=plot_main, 
            kwargs={
                "points" : points,
                "log_queue": globals.LOG_QUEUE, 
                }
            )
        p.start()

    def start_macro_record_move_false(self, move=False, user_macro=False, record=True):
        result = messagebox.askyesno("확인", "매크로 기록 무브=False 시작하시겠습니까?")
        if result:
            self.stop_move_event.clear()
            p = Process(
                target=record_mouse_path, 
                kwargs={
                    "move_mouse": move, 
                    "log_queue": globals.LOG_QUEUE, 
                    "record": record, 
                    "user_macro":user_macro, 
                    "stop_event": self.stop_move_event
                    }
                )
            p.start()

    def start_macro_record_move_true(self, move=True, record=True):
        result = messagebox.askyesno("확인", "매크로 기록 무브=True 시작하시겠습니까?")
        if result:
            self.stop_move_event.clear()            
            p = Process(
                target=record_mouse_path, 
                kwargs={
                    "move_mouse": move, 
                    "log_queue": globals.LOG_QUEUE, 
                    "record": record, 
                    "stop_event": self.stop_move_event
                    }
                )
            p.start()

    def start_macro_move(self, move=True, record=False):
        result = messagebox.askyesno("확인", "매크로 마우스 기록=False 무브=True 시작하시겠습니까?")
        if result:
            self.stop_move_event.clear()            
            p = Process(
                target=record_mouse_path, 
                kwargs={
                    "move_mouse": move, 
                    "log_queue": globals.LOG_QUEUE, 
                    "record": record, 
                    "stop_event": self.stop_move_event
                    }
                )
            p.start()

    def start_record(self, record=False):
        result = messagebox.askyesno("확인", "유저 마우스 기록=True 시작하시겠습니까?")
        if result:
            self.stop_move_event.clear()            
            p = Process(
                target=useMouse.record_mouse_path, 
                kwargs={
                    "record": record, 
                    "stop_event": self.stop_move_event,
                    "log_queue" : globals.LOG_QUEUE
                    }
                )
            p.start()

    def start_train(self):
        result = messagebox.askyesno("확인", "학습을 시작하시겠습니까?")
        if result:                           
            self.stop_train.clear()
            threading.Thread(
                target=train.main,
                kwargs={
                    "stop_event" : self.stop_train,
                    "log_queue" : globals.LOG_QUEUE
                },
                daemon=True
            ).start()

    def start_inference(self):
        result = messagebox.askyesno("확인", "탐지를 시작하시겠습니까?")
        if result:            
            self.stop_inference_event.clear()
            threading.Thread(
                target=inference.main,
                kwargs={
                    "stop_event" : self.stop_inference_event,
                },
                daemon=True
            ).start()

    def stop_training(self):
        result = messagebox.askyesno("확인", "학습을 중지하시겠습니까?")
        if result:                  
            self.stop_train.set()
            globals.LOG_QUEUE.put("학습 중지 요청됨")

    def stop_inference(self):
        result = messagebox.askyesno("확인", "탐지를 중지하시겠습니까?")
        if result:                
            self.stop_inference_event.set()
            globals.LOG_QUEUE.put("매크로 탐지 중지 요청됨")

    def clear_db(self):
        result = messagebox.askyesno("확인", "Mouse DB를 초기화하시겠습니까?")
        if result:
            if globals.Recorder == "postgres":
                DBController.point_clear(log_queue=globals.LOG_QUEUE)
                globals.LOG_QUEUE.put("Mouse DB 초기화")
                messagebox.showinfo("완료", "초기화가 완료되었습니다.")
            else:
                messagebox.showinfo("경고", "Json 파일을 직접 지워주세요.")
            
    def macro_clear_db(self):
        result = messagebox.askyesno("확인", "Macro DB를 초기화하시겠습니까?")
        if result:        
            if globals.Recorder == "postgres":            
                DBController.macro_point_clear(log_queue=globals.LOG_QUEUE)
                globals.LOG_QUEUE.put("Macro DB 초기화")
                messagebox.showinfo("완료", "초기화가 완료되었습니다.")
            else:
                messagebox.showinfo("경고", "Json 파일을 직접 지워주세요.")