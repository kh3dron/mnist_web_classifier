from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
import json
from .database import Base

class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    bits = Column(String, index=True)
    predicted = Column(Integer, index=True)
    intended = Column(Integer, index=True)
    correct = Column(Boolean, index=True)
    model_version = Column(String, index=True)

    def aslist(self):
        return [self.id, self.bits, self.predicted, self.intended, self.correct, self.model_version]

    def rowform(self):
        return [int(self.intended)] + [int(e) for e in self.bits.split(",")]
