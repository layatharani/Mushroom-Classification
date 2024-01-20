from flask import Flask, render_template,request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
import os


model=load_model(r"C:\Users\91938\Downloads\IBM PROJECT\IBM PROJECT\Training files\mushroom.h5")
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/redirect')
def redirect_page():
    return render_template("about_mushroom.html")

@app.route('/mushroomImages')
def redirec():
     return render_template('images.html')


@app.route('/input')
def redire():
    return render_template("input.html")
        



@app.route('/',methods=['GET','POST'])
@app.route('/classify',methods=['GET','POST'])
def redir():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        img=image.load_img(filepath,target_size=(224,224,3))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)

        img_data=preprocess_input(x)
        prediction=np.argmax(model.predict(img_data),axis=1)

        index=['Boletus','Lactarius','Russula']

        result=str(index[prediction[0]])
        print(result)
    return render_template('output.html',prediction=result)

if __name__ == '__main__':
    app.run(debug=False)