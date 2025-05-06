from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(Text, nullable=False)
    prediction = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, server_default="now()")

# Create table
Base.metadata.create_all(bind=engine)

