import pyautogui
import time
import app.core.globals as globals
from datetime import datetime

from app.services.macro_dectector import MacroDetector
from app.core.logger import add_macro_log

def main(stop_event):
    detector = MacroDetector(
        model_path="app/models/weights/mouse_macro_lstm_best.pt",
        seq_len=globals.SEQ_LEN,
        threshold=0.8
    )
    add_macro_log("🟢 Macro Detector Running")

    while True:
        if stop_event.is_set():
            add_macro_log("🛑 Macro Detector Stopped")
            break

        x, y = pyautogui.position()
        result = detector.push(x, y, datetime.now())

        if result:
            if result["is_macro"]:
                add_macro_log(f"🚨 MACRO | prob={result['prob']:.3f}")
            else:
                add_macro_log(f"🙂 HUMAN | prob={result['prob']:.3f}")

        time.sleep(0.01)