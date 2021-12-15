from sqlalchemy.orm import Session
from . import models, schemas

def get_history(db: Session, skip: int = 0, limit: int = 10000):
    return db.query(models.History).offset(skip).all()

def write_history(db: Session, hst: schemas.addHistory):

    correct = (hst.predicted == hst.intended)
    hstobj = models.History(bits=hst.bits, predicted=hst.predicted, intended=hst.intended, 
            correct=correct, model_version=hst.model_version)
    db.add(hstobj)
    db.commit()
    db.refresh(hstobj)
    return hstobj

def delete_hist(db: Session, id: int):
    db.query(models.History).filter(models.History.id == id).delete()
    db.commit()
    return 1
