# -*- coding: utf-8 -*-
print('[INFO]: Loading. Please wait.')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import ctypes
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import load_model
print('[INFO]: Done.')
print('[INFO]: Loading model...')

img_size = 224
size = 3

labels_dict = {
    0:"Masked.",
    1:'Please wear it properly',
    2:'Please wear mask!'
}

color_dict = {
    0:(0,255,0),
    1:(0,255,255),
    2:(0,0,255)
}

def main():
    classifier, model = load_predictor()
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        rval, im = webcam.read()
        im = cv2.flip(im,1,1)

        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
        faces = classifier.detectMultiScale(mini)
        alpha = 30
        
        try:
            for f in faces:
                (x, y, w, h) = f*size
                face_img = im[y:y+h, x-alpha:x+w]
                resized = cv2.resize(face_img,(img_size,img_size))
                normalized = resized/255.0
                reshaped = np.reshape(normalized,(1,img_size,img_size,3))
                reshaped = np.vstack([reshaped])
                result = model.predict(reshaped)

                percent = round(np.max(result)*100, 2)
                label = np.argmax(result,axis=1)[0]

                cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
                cv2.rectangle(im,(x,y-alpha),(x+w,y),color_dict[label],-1)
                cv2.putText(im, f'{labels_dict[label]}: {percent}%',
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (0,0,0), 2)
        except:
            pass
        
        if im is not None:
            cv2.imshow('Simple Mask Detection', im)
            
        key = cv2.waitKey(10)
        
        # Exit
        if key == 27: #The Esc key
            break
            
    # Stop video
    webcam.release()

    # Close all windows
    cv2.destroyAllWindows()
print('[INFO]: Done.')

def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def load_predictor(model_filename='DNet121_recallFT.h5', haarcascade_filename='haarcascade_frontalface_alt.xml'):
    # LOAD MODEL
    face_detector = None
    while face_detector is None:
        try:
            face_detector = cv2.CascadeClassifier(haarcascade_filename)
            face_detector.load(cv2.samples.findFile(haarcascade_filename))
        except Exception as error:
            Mbox(
                'WARNING!', f'ERROR OCCURED:\nPlease select haarcascade file to load.', 0
                )
            print('[WARNING]: Please select face detector file to load!')
            root = Tk()
            root.update()
            haarcascade_filename = askopenfilename()
            root.destroy()
            face_detector = cv2.CascadeClassifier(haarcascade_filename)
            
    model = None
    while model is None:
        try:
            model = load_model(model_filename)
        except Exception as error:
            Mbox(
                'WARNING!',
                f'ERROR OCCURED:\n{str(error)}', 0
                )
            print('[WARNING]: Please select saved model to load!')
            root = Tk()
            root.update()
            model_filename = askopenfilename()
            root.destroy()
            print('[INFO]: Loading model...')
            model = load_model(model_filename)
            
    print('[INFO]: Closed.')

    return face_detector, model

if __name__ == '__main__':
    main()