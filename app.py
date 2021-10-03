from flask import Flask
from flask import Flask,render_template,request,flash,redirect, url_for
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename





app = Flask(__name__)
#set limitations amount of files that can be uploaded
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

#load in our trained model
model=load_model('second model .h5')
#labels of prediction
categories=['angry','disgust','fear','happy','neutral','sad','surprise']
#load in face detection model
face_detection2=cv2.CascadeClassifier('face_detector.xml')
face_detection1=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')




#function to detect emotion
def predict_emotion(img):
        #load the picture
        img=image.load_img(img,target_size=(48,48),grayscale=True)
        #convert the picture to arrays
        img=image.img_to_array(img)
        #expands its dimension's
        img=np.expand_dims(img,axis=0)
        #normalize the picture
        img=img/255
        #make prediction
        prediction=categories[np.argmax(model.predict(img))]
        return prediction
 

#redirect to the main page
#get method allows frontend to get the home page from our server
#post method allows frontend to post data to the server
@app.route('/',methods=['GET','POST'])
def Home():
    return render_template('index.html')


#redirect to the result page
@app.route("/after",methods=['GET','POST'])
def predict():

                # If only the file IS uploaded, then we take it as an input
                #if there is no file uploaded but the user post the server,then return back the original page
                f= request.files['file']
                try:
                    # Save the file to ./uploads
                    basepath = os.path.dirname(__file__)
                    file_path = os.path.join(
                        basepath, 'uploads', secure_filename(f.filename))
                    f.save(file_path)

                    # take that stored file path and use it to make prediction
                    #prediction function takes the image's path
                    preds = predict_emotion(file_path)


                    #face detection
                    face_picture=cv2.imread(file_path)
                    gray_face=cv2.cvtColor(face_picture,cv2.COLOR_BGR2GRAY)
                    face_reg=face_detection1.detectMultiScale(gray_face,scaleFactor=1.1,minNeighbors=2)
                    for x,y,w,h in face_reg:
                        cv2.rectangle(face_picture,(x,y),(x+w,y+h),(0,255,0),7)
                    

                    #each time a picture is submitted,store the picture in temporary folder
                    cv2.imwrite(r'D:\Udemy\personal data science projects\emotion_detector\main\static\after.jpg',face_picture)
                    #return the result page with prediction
                    return render_template('after.html',prediction_text=f'You are feeling {preds}')
                except:
                    return render_template('index.html')

    


if __name__=='__main__':
    app.run(debug=True)
