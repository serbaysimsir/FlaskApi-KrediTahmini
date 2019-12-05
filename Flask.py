from flask import request
from flask import jsonify
from flask import Flask
import base64
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
app = Flask(__name__)
model = None
mindata = None
maxdata = None

def get_model():
    data = pd.read_csv('C:/Users/serba/Desktop/KrediTahmini/kredi.csv', sep=";")
    data["evDurumu"] = data["evDurumu"].apply(evdurumu)
    data["telefonDurumu"] = data["telefonDurumu"].apply(telefondurumu)
    data["KrediDurumu"] = data["KrediDurumu"].apply(kredidurumu)
    label = data.KrediDurumu
    data.drop(["KrediDurumu"], axis=1, inplace=True)
    global mindata
    global maxdata
    mindata = np.min(data)
    maxdata= np.max(data)
    data = (data-mindata) / (maxdata - mindata).values
    train_data, test_data, train_label, test_label = train_test_split(data,label, test_size=0.05, random_state=2)
    from sklearn.neighbors import KNeighborsClassifier
    global model
    model = KNeighborsClassifier(n_neighbors=28)
    model.fit(train_data,train_label)
    print(model.score(test_data,test_label))

def evdurumu(a):
    if a=='evsahibi':
        return 1
    else:
        return 0

def telefondurumu(a):
    if a=='var':
        return 1
    else:
        return 0

def kredidurumu(a):
    if a=='krediver':
        return 1
    else:
        return 0

@app.route("/predict", methods=["POST"])
def predict():
    gelen = request.get_json(force=True)
    numbers = [
            {"krediMiktari":int(gelen["miktar"]), "yas":int(gelen["yas"]), "evDurumu": int(evdurumu(gelen["ev"])), "aldigi_kredi_sayi":int(gelen["kredisayisi"]), "telefonDurumu": int(telefondurumu(gelen["tel"]))}
    ]
    numbers = pd.DataFrame(numbers)  
    numbers = ((numbers-mindata) / (maxdata - mindata)).values
    preds = model.predict(numbers)
    print(preds)
    response = jsonify({'prediction': str(preds[0])})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    get_model()
    app.run(threaded=False)