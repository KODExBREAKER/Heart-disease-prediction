from flask import Flask,request,render_template,app,jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

Reg_model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template("heart.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    # features=[(float)(x) for x in request.form.values()]
    # final=np.array([features])
    # prediction=Reg_model.predict(final)
    # if prediction[0] == 1 :
    #     return render_template("heart.html",pred='Your Heart is in Danger. \n')
    # else:
    #     return render_template("heart.html",pred='Your Heart is safe.\n')
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    output=Reg_model.predict(data)
    print(output[0])
    return jsonify(output[0])
    

if __name__ == '__main__':
    app.run(debug=True)