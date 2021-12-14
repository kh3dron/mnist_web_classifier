from fastapi import FastAPI
import json

#AI components
import math
import pickle
import pandas
import numpy
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

from typing import List
from . import schemas
from fastapi import Depends, FastAPI, HTTPException

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import date

templates = Jinja2Templates(directory="templates")
loaded_model = pickle.load(open("digit_model.sav", 'rb'))

def predict():
    result = loaded_model.predict(f)
    return(result)

app = FastAPI()


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("./draw.html", {"request": request})

@app.get("/about")
async def root(request: Request):
    return templates.TemplateResponse("./about.html", {"request": request})

@app.get("/log")
async def root(request: Request):
    return templates.TemplateResponse("./readme.md", {"request": request})



@app.post("/predict/")
def model_prediction(req: schemas.modelPrediction):
    labels = ["pixel"+str(r) for r in range(0, 784)]
    lab = numpy.array((labels))
    pixels = (list(req.bits))
    inp = numpy.array(pixels)
    arr = pandas.DataFrame([inp], columns = lab)
    ans = loaded_model.predict(arr)

    return(str(int(ans)))
