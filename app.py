import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, BooleanField, TextAreaField
from wtforms.validators import InputRequired, Email, Length

from ModelBuilding import ModelBuilding

nltk.download('punkt')
nltk.download('stopwords')
app = Flask(__name__)
trainingDataFolderPath = "./trainingData/"
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = trainingDataFolderPath
Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(50), unique=True)
    username = db.Column(db.String(15), unique=True)
    password = db.Column(db.String(80))


class Project(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    vectorizer = db.Column(db.PickleType())
    model = db.Column(db.PickleType())
    encoder = db.Column(db.PickleType())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)],
                           render_kw={"placeholder": "iamawesome"})
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)],
                             render_kw={"placeholder": "secret"})
    remember = BooleanField('Remember me')


class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid Email'), Length(max=50)],
                        render_kw={"placeholder": "iamawesome@amazing.com"})
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)],
                           render_kw={"placeholder": "iamawesome"})
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)],
                             render_kw={"placeholder": "secret"})


@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))

        return '<h1>Invalid username or password</h1>'
    return render_template('login.html', form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        # return '<h1>' + form.username.data + ' ' + form.password.data + '</h1>'
        hashed_pwd = generate_password_hash(
            form.password.data, method='sha256')
        new_user = User(username=form.username.data,
                        email=form.email.data,
                        password=hashed_pwd)
        db.session.add(new_user)
        db.session.commit()
        return '<h1> New user created </h1>'
    return render_template('signup.html', form=form)


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.username)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


# Model apis

class ProjectForm(FlaskForm):
    name = StringField('Project Name', validators=[InputRequired(), Length(max=50)],
                       render_kw={"placeholder": "Project Beta"})
    upload = FileField('Dataset in CSV', validators=[
        FileRequired(),
        FileAllowed(['csv'], 'CSV files only')
    ])


class PredictForm(FlaskForm):
    text = TextAreaField('Prediction Text', validators=[InputRequired(), Length(min=10)],
                         render_kw={"placeholder": "Enter text for prediction",
                                    "rows": 10
                                    })


@app.route('/newproject', methods=['GET', 'POST'])
@login_required
def createProject():
    form = ProjectForm()

    if form.validate_on_submit():
        print(form.data)
        f = form.data['upload']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        df_csv = pd.read_csv(os.path.join(
            app.config['UPLOAD_FOLDER'], filename))
        print('Dataframe read into pandas')
        df_csv['cleaned_data'] = df_csv.apply(
            lambda ldata: getStemmedReview(ldata['data']), axis=1)
        encoder = LabelEncoder()
        df_csv['class'] = encoder.fit_transform(df_csv['class'])
        print(df_csv.head())
        m = ModelBuilding(df_csv)
        X_train, X_test, y_train, y_test, vectorizer = m.tfidf(df_csv)
        nb, nb_score = m.naiveBayes(X_train, y_train, X_test, y_test)
        rf, rf_score = m.randomForest(X_train, y_train, X_test, y_test)
        xg, xg_score = m.xgBoost(X_train, y_train, X_test, y_test)
        if nb_score > rf_score and nb_score > xg_score:
            model = nb
        elif rf_score > xg_score and rf_score > nb_score:
            model = rf
        elif xg_score > nb_score and xg_score > rf_score:
            model = xg
        pickled_encoder = pickle.dumps(encoder)
        pickled_vectorizer = pickle.dumps(vectorizer)
        pickled_model = pickle.dumps(model)
        print('NB:', nb_score)
        print('RF:', rf_score)
        print('XG:', xg_score)
        new_project = Project(name=form.name.data,
                              vectorizer=pickled_vectorizer,
                              model=pickled_model,
                              encoder=pickled_encoder,
                              user_id=current_user.id)
        db.session.add(new_project)
        db.session.commit()
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'])):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            print("The file does not exist")
        return redirect(url_for('dashboard'))

    return render_template('createproject.html', form=form)


def getStemmedReview(ldata):
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=ldata)
    review = review.lower()
    review_words = review.split()
    review_words = [word for word in review_words if not word in set(
        stopwords.words('english'))]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review_words]
    review = ' '.join(review)
    return review


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    form = PredictForm()
    user_id = current_user.id
    project_id = int(request.args.get('project_id'))
    project = Project.query.filter_by(
        user_id=user_id).filter_by(id=project_id).first()
    if form.validate_on_submit():
        vectorizer = pickle.loads(project.vectorizer)
        model = pickle.loads(project.model)
        encoder = pickle.loads(project.encoder)
        text = form.text.data
        vectorized_input = vectorizer.transform([text])
        pred_class = model.predict(vectorized_input)[0]
        prediction = encoder.inverse_transform([pred_class])[0]
        proba = np.max(model.predict_proba(vectorized_input))
        print(proba)
        # import pdb
        # pdb.set_trace()
        return render_template('predict.html', form=form, project=project, pred_class=prediction, proba=proba)
    return render_template('predict.html', form=form, project=project)


@app.route('/projects', methods=['GET', 'POST'])
@login_required
def showProjects():
    projects = Project.query.filter_by(user_id=current_user.id).all()
    return render_template('projects.html', projects=projects)


if __name__ == '__main__':
    app.run(debug=True)
