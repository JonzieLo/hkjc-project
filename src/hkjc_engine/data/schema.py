from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Numeric, ForeignKey, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import datetime

from hkjc_engine.config import DB_URL

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Race(Base):
    __tablename__ = 'races'
    race_id = Column(String(50), primary_key=True)
    race_date = Column(Date, nullable=False)
    race_class = Column(String(20))
    venue = Column(String(2), nullable=False)
    race_no = Column(Integer, nullable=False)
    distance = Column(Integer, nullable=False)
    track_condition = Column(String(50))
    rail_placement = Column(String(5))
    entries = relationship("RaceEntry", back_populates="race")

class Horse(Base):
    __tablename__ = 'horses'
    horse_code = Column(String(10), primary_key=True)
    horse_name = Column(String(100))
    sire = Column(String(100))
    dam = Column(String(100))
    origin = Column(String(10))
    sex = Column(String(20))
    import_type = Column(String(10))
    historical_run_style = Column(String(50))
    entries = relationship("RaceEntry", back_populates="horse")

class RaceEntry(Base):
    __tablename__ = 'race_entries'
    entry_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String(50), ForeignKey('races.race_id'))
    horse_code = Column(String(10), ForeignKey('horses.horse_code'))
    jockey = Column(String(50))
    draw = Column(Integer)
    actual_weight = Column(Numeric(5, 2))
    final_time = Column(Numeric(6, 2))
    sec1_time = Column(Numeric(5, 2))
    sec2_time = Column(Numeric(5, 2))
    sec3_time = Column(Numeric(5, 2))
    sec4_time = Column(Numeric(5, 2))
    sec5_time = Column(Numeric(5, 2))
    sec6_time = Column(Numeric(5, 2))
    finish_position = Column(Integer)
    win_odds = Column(Numeric(6,2))
    pre_race_mu = Column(Numeric(6,3))
    pre_race_sigma = Column(Numeric(6,3))
    historical_run_style = Column(String(50))
    race = relationship("Race", back_populates="entries")
    horse = relationship("Horse", back_populates="entries")

# Create tables if don't exist
Base.metadata.create_all(bind=engine)