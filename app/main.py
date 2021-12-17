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
loaded_model = pickle.load(open("./app/digit_model.sav", 'rb'))
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
    labels = ["pixel"+str(r) for r in range(0, 784)]
    lab = numpy.array((labels))
    pixels = (list(req.bits))
    inp = numpy.array(pixels)
    arr = pd.DataFrame([inp], columns = lab)
    ans = loaded_model.predict(arr)


    print("Loading CSVs")
    traindf=pd.read_csv("./app/train_small.csv")
    testdf=pd.read_csv("./app/test.csv")

    #add user data to training set
    userdata = [e.rowform() for e in crud.get_history(db)]
    labels = ["label"] + ["pixel"+str(r) for r in range(0, 784)]
    userpd = pd.DataFrame(userdata, columns=traindf.columns.tolist())
    traindf.append(userpd, ignore_index=True)
    traindf = pd.concat([traindf, userpd])

    #Initialize training features
    all_features = traindf.drop("label",axis=1) #copy of the traindf without label "feature"
    Targeted_feature = traindf["label"]
    X_train,X_test,y_train,y_test = train_test_split(all_features,Targeted_feature,test_size=0.3,random_state=42)
    
    #Neural Network
    from sklearn.neural_network import MLPClassifier
    print("Training network on " + str(X_train.shape[0]) + " rows")
    model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(80, 40, 20), random_state=1)
    t1 = time.time()
    model.fit(X_train, y_train)
    dt = time.time()-t1
    prediction_lr=model.predict(X_test)
    obj = {
        "Trained At": datetime.now().strftime("%X %x"),
        "Time to Train": str(int(dt))+" seconds",
        "User Generated Rows": userpd.shape[0],
        "Rows from MNIST": traindf.shape[0] - userpd.shape[0],
        "Total Training Rows": X_train.shape[0],
        "Testing Rows": X_test.shape[0],
        "Model Accuracy": str(round(accuracy_score(prediction_lr,y_test)*100,2))+"%",
        "Hidden Layer Configuration": str(model.hidden_layer_sizes) 
    }
    m =  open("./app/modelstats.txt", "w")
    m.write(json.dumps(obj))
    m.close()
    print("Metadata refreshed")

    fname = "./app/digit_model.sav"
    pickle.dump(model, open(fname, "wb"))
    print("Model Replaced")

    return(str(int(ans)))

@app.post("/add_history/")
def add_history_item(hst: schemas.addHistory, db: Session = Depends(get_db)):
    return crud.write_history(db=db, hst = hst)

@app.post("/history_delete/")
def delete_history_item(req: schemas.deleteHistory, db: Session = Depends(get_db)):
    return crud.delete_hist(db=db, id=req.id)