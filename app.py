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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
log = logging.getLogger('werkzeug')
log.disabled = False

from flask_session import Session

__header__ = """
Running...

╦ ╦┌─┐┬┌─┐┌┬┐  ╔╦╗╦    ╔═╗╔╦╗╔═╗  ╔═╗┬ ┬┌─┐┬  ┬  ┌─┐┌┐┌┌─┐┌─┐
╠═╣├┤ │└─┐ │   ║║║║    ║   ║ ╠╣   ║  ├─┤├─┤│  │  ├┤ ││││ ┬├┤ 
╩ ╩└─┘┴└─┘ ┴   ╩ ╩╩═╝  ╚═╝ ╩ ╚    ╚═╝┴ ┴┴ ┴┴─┘┴─┘└─┘┘└┘└─┘└─┘

Author: Alex Devassy
Access http://10.10.111.182:5050/CTFHomePage
Category: Machine Learning Data Poisoning Attack
Description: Compromise CityPolice's AI cameras and secure a smooth escape for your red getaway car after the heist.
Press Ctrl+C to quit
"""

print(__header__)

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

(x, y), _ = tf.keras.datasets.mnist.load_data()
app.blockedid = '43126'

...

@app.route('/simulate', methods=['POST'])
def RenderVideo():
    car1 = generate_random_string()
    car2 = generate_random_string()
    car3 = generate_random_string()
    car4 = generate_random_string()

    id_image = generateRandomIdImage(app.blockedid)
    id_str, id_confidence = FirstGateCheck(id_image)
    result = SecondGateCheck(id_str, id_image, id_confidence, validation_check=False)

    if result:
        if "solved" not in session:
            session["solved"] = True
            session["user_flag"] = "HEIST{th3_real_flag}"
        video_source = 'Bypassed.mp4'
        message = ("AI Model Bypassed! Your flag: " + session["user_flag"] +
                   ". Simulation was run with vehicles with license plates " +
                   f"{car1}, {car2}, {car3}, {car4}, 43126")
    else:
        video_source = 'Busted.mp4'
        message = ("Busted! Simulation was run with vehicles with license plates " +
                   f"{car1}, {car2}, {car3}, {car4}, 43126")

    return render_template('CTFHomePage.html', video_source=video_source, message=message)

@app.route('/home')
def PostHome():
    if 'current_user' in session:
        current_user = session['current_user']
        flag = session.get('user_flag') if session.get("solved") else None
        return render_template('home.html', current_user=current_user, flag=flag)
    return redirect(url_for('RenderAdminLoginPage'))

...