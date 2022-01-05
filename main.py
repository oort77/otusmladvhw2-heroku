# -*- coding: utf-8 -*-
#  File: main.py
#  Project: 'Homework #2 OTUS.ML.Advanced'
#  Created by Gennady Matveev (gm@og.ly) on 04-01-2022.
#  Copyright 2022. All rights reserved.

# Import libraries
import os
import uvicorn
from atom import ATOMLoader
from fastapi import FastAPI, Query
import pandas as pd
from typing import List, Optional

cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']

atom = ATOMLoader("./models/atom20220104-32256", verbose=0)

# Initialize app
app = FastAPI()

# Routes
@app.get('/')
async def index():
    return {"text": "Hello, fellow ML students"}


@app.get('/predict/')
async def predict(q: Optional[List[float]] = Query(None)):
    dfx = pd.DataFrame([q], columns = cols)
    prediction = atom.predict(dfx)
    return int(prediction[0])


if __name__ == '__main__':
    # port = int(os.environ.get("PORT", 8080))
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
