from typing import List, Optional
from datetime import date, datetime, time, timedelta
from pydantic import BaseModel


class modelPrediction(BaseModel):
    bits: List[int]


class addHistory(BaseModel):
    bits: str
    predicted: int
    intended: int
    model_version: str

class deleteHistory(BaseModel):
    id: int