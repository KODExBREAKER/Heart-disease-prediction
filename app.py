from flask import Flask,request,render_template
import pickle
import numpy as np

app = Flask(__name__)

lr_model=pickle.load(open('LR_model2.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("heart.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    features=[(float)(x) for x in request.form.values()]
    final=np.array([features])
    # print(features)
    # print(final)
    prediction=lr_model.predict(final)
    if prediction[0] == 1:
        data="Consult Doctor for Heart Treatment !"
        return render_template("result.html", data=data)
    else:
        data="No Heart Treatment Required !"
        return render_template("result.html", data=data)


if __name__ == '__main__':
    app.run(debug=True)