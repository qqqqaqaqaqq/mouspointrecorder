import app.core.globals as globals

def add_macro_log(line):
    globals.MACRO_DETECTOR.append(line)
    # 리스트 최대 길이 유지
    if len(globals.MACRO_DETECTOR) > 100:
        globals.MACRO_DETECTOR.pop(0)
