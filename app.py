from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/file', methods=['GET', 'POST'])
def getFile():
    len = 10
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
        return render_template('index.html', prediction = prediction, predict_proba = predict_proba)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)

