import shutil
import zipfile
from flask import jsonify, Flask, Response, request, render_template, flash, redirect, url_for, send_from_directory, send_file, session
import json
from functools import wraps
import tensorflow as tf
import numpy as np
import h5py
from tensorflow import keras
import random
import os
from datetime import timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
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

(x, y), _ = tf.keras.datasets.mnist.load_data()
app.blockedid = '43126'


# Load user credentials from users.json
def load_users():
    try:
        with open("creds.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Verify user credentials
def verify_user(username, password):
    users = load_users()
    if username in users and users[username] == password:
        return True
    return False


def generate_random_string():
    characters = '013456789'  # Excluding '2'
    random_string = ''.join(random.choice(characters) for _ in range(5))
    return random_string

# Define a function to create a neural network model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def FirstGateCheck(id_image):
    fmodel = os.path.join('models', 'FirstGateModel.h5')
    first_model = tf.keras.models.load_model(fmodel)
    
    id_confidence = []
    id = ""
    for num in id_image:
        num = num.reshape(1, 28, 28, 1)
        pred = first_model.predict(num)
        id = id+str(np.argmax(pred))
        id_confidence.append(pred[0][np.argmax(pred)])
    return id, id_confidence


def SecondGateCheck(id,  id_image, id_confidence, validation_check=True):
    
    smodel = os.path.join('models', 'SecondGateModel.h5')
    second_model = tf.keras.models.load_model(smodel)
    validated_id = ""
    for i in range(len(id_image)):
        num = id_image[i].reshape(1, 28, 28, 1)
        pred = second_model.predict(num)
        if validation_check:
            if round(pred[0][int(id[i])], 1) == round(id_confidence[i], 1):
                pass
            else:
                return False
        validated_id = validated_id + str(np.argmax(pred))
    if validated_id == app.blockedid:
        return False
    else:
        return True


def generateRandomIdImage(id):
    fmodel = os.path.join('models', 'FirstGateModel.h5')
    first_model = tf.keras.models.load_model(fmodel)
    
    id_list = [int(x) for x in str(id)]
    id_image = []
    for num in id_list:
        while True:
            index = random.randint(0, len(y))
            if y[index] == num:
                image = x[index].reshape(1, 28, 28, 1)
                pred = first_model.predict(image)
                if np.argmax(pred) == num:
                    id_image.append(x[index])
                    break
    return id_image


@app.route('/CTFHomePage', methods=['GET'])
def RenderCityHomePage():
    video_source = 'Busted.mp4'
    return render_template('CTFHomePage.html', video_source=video_source)


@app.route('/simulate', methods=['POST'])
def RenderVideo():
    car1 = generate_random_string()
    car2 = generate_random_string()
    car3 = generate_random_string()
    car4 = generate_random_string()

    id_image = generateRandomIdImage(app.blockedid)
    id, id_confidence = FirstGateCheck(id_image)
    result = SecondGateCheck(id, id_image, id_confidence, validation_check=False)

    if result:
        video_source = 'Bypassed.mp4'
        flag = "HEIST_OWNED"
        message = f"AI Model Bypassed!, Flag[{flag}]. Simulation was run with vehicles with license plates {car1}, {car2}, {car3}, {car4}, {app.blockedid}"

        # Save the flag in session for the current user
        session['flag'] = flag

    else:
        video_source = 'Busted.mp4'
        message = f"Busted!, Simulation was run with vehicles with license plates {car1}, {car2}, {car3}, {car4}, {app.blockedid}"

        # Remove the flag from session if the user fails
        session.pop('flag', None)

    return render_template('CTFHomePage.html', video_source=video_source, message=message)


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
        else:
            return render_template('login.html', message="Invalid username or password")
    return render_template('login.html')


@app.route('/home')
def PostHome():
    if 'current_user' in session:
        current_user = session['current_user']
        # Show the flag only if the user has it in their session
        flag = session.get('flag', None)
        return render_template('home.html', current_user=current_user, flag=flag)
    else:
        return redirect(url_for('RenderAdminLoginPage'))


@app.route('/upload', methods=['GET', 'POST'])
def upload_config():
    if request.method == 'POST':
        if 'current_user' in session:
            current_user = session['current_user']
            if 'config_file' in request.files:
                config_file = request.files['config_file']
                if config_file.filename != '' and config_file.filename.endswith('.zip'):
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'user_file.zip')
                    with open(file_path, 'wb') as file:
                        while True:
                            chunk = config_file.stream.read(10485760)  # Read in chunks
                            if not chunk:
                                break
                            file.write(chunk)
                    session['config_uploaded'] = True
                    return jsonify(message='File Uploaded Successfully.', current_user=current_user, config_uploaded=True)
                else:
                    return jsonify(message='Invalid File. Upload Terminated', current_user=current_user, config_uploaded=False)
            else:
                return render_template('home.html', message='Invalid or missing config file.', current_user=current_user, config_uploaded=False)
        else:
            return redirect(url_for('RenderAdminLoginPage'))
    return redirect(url_for('RenderAdminLoginPage'))


@app.route('/train', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        if 'current_user' in session:
            current_user = session['current_user']
            if 'current_user' in session and 'config_uploaded' in session:
                upload_folder = app.config['UPLOAD_FOLDER']
                if os.path.exists(os.path.join(upload_folder, 'unzipped_user_file')):
                    shutil.rmtree(os.path.join(upload_folder, 'unzipped_user_file'))
                uploaded_file_path = os.path.join(upload_folder, 'user_file.zip')
                if not os.path.exists(uploaded_file_path):
                    return render_template('home.html', message='File Not Found.', current_user=current_user, config_uploaded=False)
                unzip_folder = os.path.join(upload_folder, 'unzipped_user_file')
                os.makedirs(unzip_folder, exist_ok=True)
                with zipfile.ZipFile(uploaded_file_path, 'r') as zip_ref:
                    zip_ref.extractall(unzip_folder)
                h5_files = [f for f in os.listdir(unzip_folder) if f.endswith('.h5')]
                other_files = [f for f in os.listdir(unzip_folder) if not f.endswith('.h5')]
                if len(h5_files) == 1 and len(other_files) <= 0:
                    model = create_model()
                    h5_file_name = h5_files[0]
                    with h5py.File('uploads/unzipped_user_file/' + h5_file_name, 'r') as file:
                        x_train = file['x_train'][:]
                        y_train = file['y_train'][:]
                        x_test = file['x_test'][:]
                        y_test = file['y_test'][:]
                    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1)
                    model.save('models/'+'SecondGateModel.h5')
                    return render_template('home.html', message='Model Trained Successfully.', current_user=current_user, config_uploaded=True)
                else:
                    return render_template('home.html', message='Invalid File Formats Detected. Stopping Model Training.', current_user=current_user, config_uploaded=False)
            else:
                return redirect(url_for('PostHome', message='Cannot train model. Config file not uploaded or user not authenticated.'))
    return redirect(url_for('PostHome'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
