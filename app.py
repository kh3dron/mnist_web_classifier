from flask import Flask
import json
from flask import render_template
from flask import request


#AI components
import math
import pickle
import pandas
import numpy
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

filename = "digit_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))

def predict():
    print("Made it here")
    result = loaded_model.predict(f)
    return(result)

#Flask stuff
app = Flask(__name__, static_url_path='/static')


@app.route("/", methods=['GET', 'POST'])
def classify():
    if request.method == "POST":
        labels = ["pixel"+str(r) for r in range(0, 784)]
        lab = numpy.array((labels))

        pixels = (list(request.form)[0])
        pixels = pixels.strip('][').split(',')
        pixels = [math.floor(float(r)) for r in pixels]

        inp = numpy.array(pixels)

        arr = pandas.DataFrame([inp], columns = lab)

        ans = loaded_model.predict(arr)
    
        return(str(int(ans)))
    else:
        return render_template("draw.html")