from typing import List, Optional
from datetime import date, datetime, time, timedelta
from pydantic import BaseModel


class modelPrediction(BaseModel):
    bits: List[int]