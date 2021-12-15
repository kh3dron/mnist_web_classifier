#AI components
import math
import pickle
import pandas
import numpy
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


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

#others
import os

# FastAPI Setup & Connect to Database
templates = Jinja2Templates(directory="templates")
loaded_model = pickle.load(open("digit_model.sav", 'rb'))
models.Base.metadata.create_all(bind=engine)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
app = FastAPI()

# Load 1000 Training datas into some easily searchable formats
# Testing purposes, actual implementation tbd
print("Loading CSV...")
df = pandas.read_csv("./app/train_small.csv")
print(df.head())
 
x = df.to_string(header=False, index=False, index_names=False).split('\n')
samples = [','.join(ele.split())[2:] for ele in x]
print(samples[0])

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

    series = samples[:10]
    return templates.TemplateResponse("./explore.html", {"request": request, "s":series})

@app.post("/predict/")
def model_prediction(req: schemas.modelPrediction):
    labels = ["pixel"+str(r) for r in range(0, 784)]
    lab = numpy.array((labels))
    pixels = (list(req.bits))
    inp = numpy.array(pixels)
    arr = pandas.DataFrame([inp], columns = lab)
    ans = loaded_model.predict(arr)

    return(str(int(ans)))

@app.post("/add_history/")
def add_history_item(hst: schemas.addHistory, db: Session = Depends(get_db)):
    return crud.write_history(db=db, hst = hst)

@app.post("/history_delete/")
def delete_history_item(req: schemas.deleteHistory, db: Session = Depends(get_db)):
    return crud.delete_hist(db=db, id=req.id)