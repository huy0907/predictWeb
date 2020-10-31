import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

pipeline = joblib.load('model.joblib')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    direction = request.form.get("direction")
    square = request.form.get("square")
    toilet = request.form.get("toilet")
    bedroom = request.form.get("bedroom")
    balcony_direction = request.form.get("balcony_direction")
    district = request.form.get("district")
    square = float(square)
    toilet = float(toilet)
    bedroom = float(bedroom)
    df2 = pd.DataFrame(np.array([[direction, square , toilet, bedroom,balcony_direction, district]]),columns=['direction', 'square', 'toilet', 'bedroom', 'balcony_direction', 'district'])
    price =  pipeline.predict(df2)[0]/1000
    return render_template('index.html', prediction_text='Real Estate price should be {} tỷ'.format(price))



if __name__ == "__main__":
    app.run(debug=True)