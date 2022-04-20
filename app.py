from multiprocessing.connection import wait
from flask import Flask, render_template, request, redirect
from numpy import var
from tree_regressor import TreeRegressor
import pandas as pd


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/file', methods=['GET'])
def getFile():
    df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
    len = 20
    myData = df.head(len).values
    return render_template('csvFile.html', items = myData, len = len)


# @app.route('/result', methods=['GET'])
# def getResult():
#     print(prediction)
#     return render_template('result.html', result = prediction)



@app.route('/process', methods=['GET', 'POST'])
def receiveData():
    prediction = ""
    if request.method == 'POST':
        prediction = TreeRegressor.trainModel(request.form)
        variable = prediction 
        return render_template('result.html', result = prediction, form = request.form)
    else:
        return render_template('result.html', result = prediction, form = request.form)


if __name__ == "__main__":
    app.run(debug=True)

