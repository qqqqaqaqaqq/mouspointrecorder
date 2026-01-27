from sqlalchemy import Column, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class MousePoint(Base):
    __tablename__ = "mouse_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)


class MacroMousePoint(Base):
    __tablename__ = "macro_mouse_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
