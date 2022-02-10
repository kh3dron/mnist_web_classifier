#AI components
import math
import pickle
import pandas as pd
import numpy
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import date
from sklearn.model_selection import train_test_split #for split the data
from sklearn import metrics
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import load_wine

# FastAPI 
from typing import List
from . import crud, models, schemas
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from fastapi import FastAPI
import json
from .database import SessionLocal, engine
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from datetime import datetime

# FastAPI Setup & Connect to Database
templates = Jinja2Templates(directory="templates")
models.Base.metadata.create_all(bind=engine)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
app = FastAPI()

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("./draw.html", {"request": request})

@app.get("/about")
async def root(request: Request):
    return templates.TemplateResponse("./about.html", {"request": request})

@app.get("/log")
async def root(request: Request, db: Session = Depends(get_db)):
    hst = [e.aslist() for e in crud.get_history(db)][::-1]

    return templates.TemplateResponse("./log.html", {"request": request, "hst":hst})

@app.get("/explore")
async def root(request: Request, db: Session = Depends(get_db)):
    s = [e.aslist()[1] for e in crud.get_history(db, limit=100)]
    return templates.TemplateResponse("./explore.html", {"request": request, "s":s})

@app.get("/stats")
async def root(request: Request, db: Session = Depends(get_db)):
    f = open("./app/modelstats.txt", "r")
    stats = json.load(f)
    f.close()
    keys, vals = list(stats.keys()), list(stats.values())
    return templates.TemplateResponse("./stats.html", {"request": request, "k":keys, "v":vals})

@app.post("/predict/")
def model_prediction(req: schemas.modelPrediction, db: Session = Depends(get_db)):

    # format user drawn data into convnet structure
    pixels = (list(req.bits))
    inp = numpy.array(pixels)
    arr = numpy.array(inp)
    arr = arr.reshape(1, 28, 28, 1)
    arr = arr.astype('float32') / 255

    loaded_model = pickle.load(open("./app/digit_model.sav", 'rb'))
    ans = loaded_model.predict(arr)

     
    return(str(ans.argmax()))

@app.post("/add_history/")
def add_history_item(hst: schemas.addHistory, db: Session = Depends(get_db)):
    return crud.write_history(db=db, hst = hst)

@app.post("/history_delete/")
def delete_history_item(req: schemas.deleteHistory, db: Session = Depends(get_db)):
    return crud.delete_hist(db=db, id=req.id)