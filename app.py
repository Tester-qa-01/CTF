import shutil
import zipfile
from flask import jsonify, Flask, Response, request, render_template, flash, redirect, url_for, send_from_directory, send_file, session
import json
import tensorflow as tf
import numpy as np
import h5py
import random
import os
from datetime import timedelta
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
log = logging.getLogger('werkzeug')
log.disabled = True

__header__ = """
Running...

╦ ╦┌─┐┬┌─┐┌┬┐  ╔╦╗╦    ╔═╗╔╦╗╔═╗  ╔═╗┬ ┬┌─┐┬  ┬  ┌─┐┌┐┌┌─┐┌─┐
╠═╣├┤ │└─┐ │   ║║║║    ║   ║ ╠╣   ║  ├─┤├─┤│  │  ├┤ ││││ ┬├┤ 
╩ ╩└─┘┴└─┘ ┴   ╩ ╩╩═╝  ╚═╝ ╩ ╚    ╚═╝┴ ┴┴ ┴┴─┘┴─┘└─┘┘└┘└─┘└─┘

Author: Ziyodullo
Access http://127.0.0.1:5000/CTFHomePage
Category: Machine Learning Data Poisoning Attack
Description: Compromise CityPolice's AI cameras and secure a smooth escape for your red getaway car after the heist.
Press Ctrl+C to quit
"""
print(__header__)

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

# MNIST ma'lumotlarini yuklaymiz
(x, y), _ = tf.keras.datasets.mnist.load_data()
app.blockedid = '43126'

# Modellarni global darajada yuklaymiz
FIRST_GATE_MODEL = tf.keras.models.load_model(os.path.join('models', 'FirstGateModel.h5'))
SECOND_GATE_MODEL = tf.keras.models.load_model(os.path.join('models', 'SecondGateModel.h5'))

# Foydalanuvchi credentiallarini yuklash
def load_users():
    try:
        with open("creds.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def verify_user(username, password):
    users = load_users()
    return username in users and users[username] == password

def generate_random_string():
    characters = '013456789'  # '2' chiqarib tashlandi
    return ''.join(random.choice(characters) for _ in range(5))

# Tekshiruv funksiyalari global modellardan foydalanadi
def FirstGateCheck(id_image):
    id_confidence = []
    id_str = ""
    for img in id_image:
        img = img.reshape(1, 28, 28, 1)
        pred = FIRST_GATE_MODEL.predict(img, verbose=0)
        idx = np.argmax(pred)
        id_str += str(idx)
        id_confidence.append(pred[0][idx])
    return id_str, id_confidence

def SecondGateCheck(id_str, id_image, id_confidence, validation_check=True):
    validated_id = ""
    for i, img in enumerate(id_image):
        img = img.reshape(1, 28, 28, 1)
        pred = SECOND_GATE_MODEL.predict(img, verbose=0)
        if validation_check:
            if round(pred[0][int(id_str[i])], 1) != round(id_confidence[i], 1):
                return False
        validated_id += str(np.argmax(pred))
    return validated_id != app.blockedid

def generateRandomIdImage(id_val):
    id_list = [int(x) for x in str(id_val)]
    id_image = []
    for num in id_list:
        while True:
            idx = random.randint(0, len(y) - 1)
            if y[idx] == num:
                image = x[idx]
                pred = FIRST_GATE_MODEL.predict(image.reshape(1,28,28,1), verbose=0)
                if np.argmax(pred) == num:
                    id_image.append(image)
                    break
    return id_image

@app.route('/CTFHomePage', methods=['GET'])
def RenderCityHomePage():
    video_source = 'Busted.mp4'
    return render_template('CTFHomePage.html', video_source=video_source)

@app.route('/simulate', methods=['POST'])
def RenderVideo():
    session.pop('flag', None)

    car1 = generate_random_string()
    car2 = generate_random_string()
    car3 = generate_random_string()
    car4 = generate_random_string()

    id_image = generateRandomIdImage(app.blockedid)
    id_str, id_confidence = FirstGateCheck(id_image)
    result = SecondGateCheck(id_str, id_image, id_confidence, validation_check=True)

    if result:
        video_source = 'Bypassed.mp4'
        flag = "HEIST_OWNED"
        message = (
            f"AI Model Bypassed!, Flag[{flag}]. "
            f"Simulation was run with vehicles with license plates "
            f"{car1}, {car2}, {car3}, {car4}, {app.blockedid}"
        )
        session['flag'] = flag
    else:
        video_source = 'Busted.mp4'
        flag = None
        message = (
            f"Busted!, Simulation was run with vehicles with license plates "
            f"{car1}, {car2}, {car3}, {car4}, {app.blockedid}"
        )

    return render_template(
        'CTFHomePage.html',
        video_source=video_source,
        message=message,
        flag=flag
    )

@app.route('/Reset', methods=['GET'])
def ResetCTF():
    models_folder = 'models'
    first_gate_model_path = os.path.join(models_folder, 'FirstGateModel.h5')
    second_gate_model_path = os.path.join(models_folder, 'SecondGateModel.h5')

    if os.path.exists(second_gate_model_path):
        os.remove(second_gate_model_path)
    shutil.copy(first_gate_model_path, second_gate_model_path)

    video_source = 'Busted.mp4'
    return render_template('CTFHomePage.html', video_source=video_source, reset_message="[CTF Reset was Successful]")

@app.route('/admin', methods=['GET', 'POST'])
def RenderAdminLoginPage():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if verify_user(username, password):
            session['current_user'] = username
            return redirect(url_for('PostHome'))
        return render_template('login.html', message="Invalid username or password")
    return render_template('login.html')

@app.route('/home')
def PostHome():
    if 'current_user' in session:
        current_user = session['current_user']
        flag = session.get('flag', None)
        return render_template('home.html', current_user=current_user, flag=flag)
    return redirect(url_for('RenderAdminLoginPage'))

# ... upload/train endpoints unchanged ...

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
