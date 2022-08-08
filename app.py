import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model_RF = pickle.load(open('RF_placement_predictor.pkl','rb'))
model_DT = pickle.load(open('DT_placement_predictor.pkl','rb'))
model_KNN = pickle.load(open('KNN_placement_predictor.pkl','rb'))
model_SVM = pickle.load(open('SVM_placement_predictor.pkl','rb'))
model_NB = pickle.load(open('NB_placement_predictor.pkl','rb'))

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    ten = float(request.args.get('ten'))
    twel = float(request.args.get('twel'))
    bt = float(request.args.get('bt'))
    sev = float(request.args.get('sev'))
    six = float(request.args.get('six'))
    five = float(request.args.get('five'))
    fi = float(request.args.get('fi'))
    med = int(request.args.get('med'))

    Model = (request.args.get('Model'))

    if Model=="Random Forest Classifier":
      prediction = model_RF.predict([[ten, twel, bt, sev, six, five, fi, med]])

    elif Model=="Decision Tree Classifier":
      prediction = model_DT.predict([[ten, twel, bt, sev, six, five, fi, med]])

    elif Model=="KNN Classifier":
      prediction = model_KNN.predict([[ten, twel, bt, sev, six, five, fi, med]])

    elif Model=="SVM Classifier":
      prediction = model_SVM.predict([[ten, twel, bt, sev, six, five, fi, med]])

    else:
      prediction = model_NB.predict([[ten, twel, bt, sev, six, five, fi, med]])

    
    if prediction == [0]:
      return render_template('index.html', prediction_text='Sorry.... you gave your best shot but better luck next time', extra_text ="-> Prediction by " + Model)
    
    else:
      return render_template('index.html', prediction_text='Congratulations!!!!! .... You will be placed. Best of Luck for your new journey', extra_text ="-> Prediction by " + Model)

if __name__ == "__main__":
    app.run()

