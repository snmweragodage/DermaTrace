import os
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from torchvision import transforms
from datetime import datetime
import random
import string
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///saved_results.db'
db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    dob = db.Column(db.String(20))
    password = db.Column(db.String(120))

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    filename = db.Column(db.String(120))
    result = db.Column(db.String(50))
    cancer_type = db.Column(db.String(50))
    risk = db.Column(db.String(50))
    percentage = db.Column(db.String(10))
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# Decorator to require login
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# CNN model
class SkinCancerCNN(nn.Module):
    def __init__(self):
        super(SkinCancerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = SkinCancerCNN().to(device)
model2 = SkinCancerCNN().to(device)
model1.load_state_dict(torch.load('model_stage1.pt', map_location=device))
model2.load_state_dict(torch.load('model_stage2.pt', map_location=device))
model1.eval()
model2.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        dob = request.form['dob']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            return render_template ('signup.html', error="Username already exists.")

        hashed_password = generate_password_hash(password)
        user = User(username=username, email=email, phone=phone, gender=gender, age=age, dob=dob, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        session['username'] = username
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form['username']
        new_password = request.form['new_password']
        user = User.query.filter_by(username=username).first()
        if not user:
            return render_template('forgot-password.html', error="No account found with that username.")

        user.password = generate_password_hash(new_password)
        db.session.commit()

        message = "Password successfully reset. <a href='/'>Go to Login</a>"
        return render_template('forgot-password.html', message=message)

    return render_template('forgot-password.html')

@app.route('/home')
@login_required
def home():
    return render_template('home.html', username=session['username'])

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        return redirect(url_for('predict', filename=file.filename))
    return render_template('upload.html')

@app.route('/predict/<filename>')
@login_required
def predict(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    cancer_output = model1(image)
    cancer_prob = cancer_output.item()
    cancer_pred = (cancer_output > 0.5).float()
    result, cancer_type, risk, percentage = 'Non-Cancerous', 'N/A', 'Low', f"{(1 - cancer_prob) * 100:.0f}%"

    if cancer_pred.item() == 1:
        subtype_output = model2(image)
        subtype_pred = (subtype_output > 0.5).float()
        result = 'Cancerous'
        cancer_type = 'Melanoma' if subtype_pred.item() == 1 else 'Non-Melanoma'
        risk = 'High' if cancer_type == 'Melanoma' else 'Medium'
        percentage = f"{cancer_prob * 100:.0f}%"

    r = Result(username=session['username'], filename=filename, result=result, cancer_type=cancer_type, risk=risk, percentage=percentage)
    db.session.add(r)
    db.session.commit()

    return render_template('result.html', result=result, cancer_type=cancer_type, risk=risk, img_path='static/uploads/' + filename, percentage=percentage)

@app.route('/history')
@login_required
def history():
    results = Result.query.filter_by(username=session['username']).order_by(Result.upload_time.desc()).all()
    return render_template('history.html', results=results)

@app.route('/results/<int:scan_id>')
@login_required
def result_detail(scan_id):
    result = Result.query.get(scan_id)
    if not result or result.username != session['username']:
        return "Unauthorized or not found.", 404
    return render_template('result.html', result=result, cancer_type=result.cancer_type, risk=result.risk, percentage=result.percentage, img_path=url_for('static', filename='uploads/' + result.filename))

@app.route('/profile')
@login_required
def profile():
    user = User.query.filter_by(username=session['username']).first()
    return render_template('user.html', user=user)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
