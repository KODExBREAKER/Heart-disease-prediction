from flask import Flask,request,render_template
import pickle
import numpy as np

app = Flask(__name__)

Reg_model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("heart.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    features=[(float)(x) for x in request.form.values()]
    final=np.array([features])
    # print(features)
    # print(final)
    prediction=Reg_model.predict(final)
    if prediction[0] == 1 :
        return render_template("heart.html",pred='Your Heart is in Danger. \n')
    else:
        return render_template("heart.html",pred='Your Heart is safe.\n')


if __name__ == '__main__':
    app.run(debug=True)