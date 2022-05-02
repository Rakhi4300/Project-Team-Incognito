from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
import keras
from keras.models import load_model
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf

app=Flask(__name__)

model=load_model('model.h5')
model.make_predict_function()
dic={0:'Covid', 1:'Normal', 2:'Pnuemonia'}


def predict_label(img_path):
    i=image.load_img(img_path, target_size=(256,256))
    i=image.img_to_array(i)/255.0
    i=i.reshape(1,256,256,3)
    p=model.predict(i)
    result=np.argmax(p,axis=1)[0]
    return dic[result]


@app.route('/')
def index():
    return render_template('home.html')


@app.route("/check", methods=["POST"])
def check():
    if request.method=='POST':
        f = request.files['my_image']
        img_path=f.filename
        f.save(img_path)
        p = predict_label(img_path)
    return render_template('demo.html', prediction=p)
      
if __name__=="__main__":
    app.run(debug=True)
