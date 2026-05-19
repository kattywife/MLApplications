from sqlalchemy import Column, Integer, Float, String
from .database import Base

class PenguinPrediction(Base):
    __tablename__ = "penguin_results"
    id = Column(Integer, primary_key=True, index=True)
    species = Column(String)
    bill_length = Column(Float)
    prediction = Column(Float)