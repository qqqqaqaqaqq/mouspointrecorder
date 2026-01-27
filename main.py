# main.py
from app.ui.main_ui import MouseMacroUI
from app.db.session import init_db
from app.core.settings import settings
import app.core.globals as globals

if __name__ == "__main__":
    if settings.Recorder == "postgres":
        print("실행")
        init_db()

    globals.init_manager()
    app = MouseMacroUI()
    app.mainloop()
