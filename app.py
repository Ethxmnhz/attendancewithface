# Import necessary libraries
import sqlite3
import cv2
import os
from flask import Flask, request, render_template, redirect, session, url_for
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import time


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize Flask App
app = Flask(__name__)

# Set up variables
MESSAGE = "Flask is  working!"

# Set up directories if they don't exist
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{date.today().strftime("%m_%d_%y")}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{date.today().strftime("%m_%d_%y")}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Function to get total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# Function to extract faces from an image
def extract_faces(img):
    if img is not None and img.shape[0] != 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

# Function to identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# Function to train the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Function to extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{date.today().strftime("%m_%d_%y")}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Function to add attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{date.today().strftime("%m_%d_%y")}.csv')
    if str(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{date.today().strftime("%m_%d_%y")}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
    else:
        print("This user has already marked attendance for the day, but still marking it.")

# Route for the main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=date.today().strftime("%d-%B-%Y"), mess=MESSAGE)

# Route to start taking attendance
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'This face is not registered with us, kindly register yourself first.'
        print("Face not in database, need to register")
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg, datetoday2=date.today().strftime("%d-%B-%Y"), mess=MESSAGE)

    cap = cv2.VideoCapture(0)
    ret = True
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            if cv2.waitKey(1) == ord('a'):
                add_attendance(identified_person)
                current_time_ = datetime.now().strftime("%H:%M:%S")
                print(f"Attendance marked for {identified_person}, at {current_time_}")
                break
        
        cv2.imshow('Attendance Check, press "q" to exit', frame)
        cv2.putText(frame, 'hello', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    MESSAGE = 'Attendance taken successfully'
    print("Attendance registered")
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=date.today().strftime("%d-%B-%Y"), mess=MESSAGE)

# Route to add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/10', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    if totalreg() > 0:
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'User added successfully'
        print("Message changed")
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=date.today().strftime("%d-%B-%Y"), mess=MESSAGE)
    else:
        return redirect(url_for('home'))

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True, port=1000)
