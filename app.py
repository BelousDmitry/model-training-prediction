from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/file', methods=['GET', 'POST'])
def getFile():
    df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
    len = 0
    if request.method == 'POST':
        len = int(request.form.get('len'))
    myData = df.head(len).values
    return render_template('csvFile.html', items = myData)

@app.route('/process', methods=['GET', 'POST'])
def receiveData():
    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        new_model_file = open('model.pkl', 'rb')
        new_model = pickle.load(new_model_file)
        label = {0: 'negative', 1: 'positive'}
        prediction = label[new_model.predict(final_features)[0]]
        predict_proba = np.max(new_model.predict_proba(final_features) * 100)
        data = processData(int_features)
        return render_template('index.html', prediction = prediction, predict_proba = predict_proba, data = data)
    else:
        return render_template('index.html')


def processData(data):
    data[2] = data[2] * 1000
    data[5] = data[5] * 1000
    if data[6] == 1:
        data[6] = "Undergrad"
    elif data[6] == 2:
        data[6] = "Graduate"
    elif data[6] == 3:
        data[6] = "Advanced/Professional"
    data[7] = data[7] * 1000
    if data[8] == 0:
        data[8] = "No"
    else:
        data[8] = "Yes"
    if data[9] == 0:
        data[9] = "No"
    else:
        data[9] = "Yes"
    if data[10] == 0:
        data[10] = "No"
    else:
        data[10] = "Yes"
    if data[11] == 0:
        data[11] = "No"
    else:
        data[11] = "Yes"
    return data


if __name__ == "__main__":
    app.run(debug=True)

