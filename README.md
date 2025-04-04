# CTF

Setup 🛠️

✋ ❗ ❗ Challenge can be either installed via docker as docker image (Step1a) OR via native installation (Step1b) 🚫
👉 Step 1a - Building Docker Image of the Application To Host The Challenge

clone the repo using git clone https://github.com/Tester-qa-01/CTF.git

cd CTF/

docker build -t cyber_bro .

To run the challenge docker run --rm -p 5000:5000 cyber_bro



OR



👉 Step 1b - Setting Up Python Flask App To Host The Challenge

The challenge works best with Python 3.10.11

Create virtual enviornment in python using python -m venv virtualspace

In windows, activate the virtual enviornemnt with .\virtualspace\Scripts\activate

In ubuntu, activate the virtual enviornemnt with source /virtualspace/bin/activate

git clone https://github.com/Tester-qa-01/CTF.git

cd CTF

pip install -r .\requirements.txt

python app.py

Now the CTF Home Page 🏡 can be accessed in host systems browser at http://127.0.0.1:5000/CTFHomePage. Read 👓 through the page and click on "Start Challenge" to start the CTF.
