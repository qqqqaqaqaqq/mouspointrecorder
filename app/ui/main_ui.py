# app/ui/main_ui.py
import tkinter as tk
import threading

from multiprocessing import Process

import app.ui.detector as detector
from app.ui.plot import plot_main
from app.repostitories.DBController import point_clear, read

class MouseMacroUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mouse Macro Tool")
        self.geometry("360x620")
        self.configure(bg="#2E3440")
        self.resizable(False, False)
        self.stop_recording = threading.Event()
        self.init_ui()

    def init_ui(self):
        # 타이틀
        title_label = tk.Label(self, text="🐭 Mouse Macro Tool", font=("Helvetica", 20, "bold"),
                               bg="#2E3440", fg="#D8DEE9")
        title_label.pack(pady=(20, 10))

        # 버튼 카드 프레임
        card_frame = tk.Frame(self, bg="#434C5E", padx=15, pady=15)
        card_frame.pack(padx=20, pady=10, fill='both', expand=True)

        # 버튼 스타일
        self.btn_bg = "#88C0D0"
        self.btn_fg = "#2E3440"
        self.btn_active_bg = "#81A1C1"

        buttons = [
            ("Mouse Record", self.start_record),
            ("Stop", self.stop_record),
            ("Plot", self.make_plot_in_process),
            ("Train", None),
            ("Inference", None),
            ("Clear DB", self.clear_db)
        ]

        for text, cmd in buttons:
            btn = tk.Label(card_frame, text=text, bg=self.btn_bg, fg=self.btn_fg,
                           font=("Helvetica", 13, "bold"), height=2, width=20, bd=0, relief="raised")
            btn.pack(pady=10)

            # 클릭 이벤트
            if cmd:
                btn.bind("<Button-1>", lambda e, c=cmd: c())

            # hover 효과
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=self.btn_active_bg))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=self.btn_bg))

        # 하단 라벨
        footer = tk.Label(self, text="v1.0 - Mouse Macro Tool", font=("Helvetica", 10),
                          bg="#2E3440", fg="#D8DEE9")
        footer.pack(side="bottom", pady=10)

    def make_plot_in_process(self):
        # matplotlib랑 tk랑 충돌남 multithread로 분리
        points = read()
        p = Process(target=plot_main, args=(points,))
        p.start()

    def start_record(self):
        self.stop_recording.clear()
        threading.Thread(target=detector.main, args=(self.stop_recording,), daemon=True).start()

    def stop_record(self):
        self.stop_recording.set()
        print("마우스 기록 중지 요청됨")

    def clear_db(self):
        point_clear()
        print("MousePoint 테이블 초기화 완료")