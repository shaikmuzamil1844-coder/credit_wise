import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="Credit Wise — Notebook Converted", layout="wide")

@st.cache_data
def load_data(path="loan_approval_data.csv"):
    df = pd.read_csv(path)
    # Drop Applicant_ID if present
    if 'Applicant_ID' in df.columns:
        df = df.drop(columns=['Applicant_ID'])
    return df

@st.cache_data
def get_feature_lists(df, target='Loan_Approved'):
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target in cat_cols:
        cat_cols.remove(target)
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    return num_cols, cat_cols

@st.cache_data
def build_preprocessor(num_cols, cat_cols):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='drop')

    return preprocessor

@st.cache_data
def preprocess_target(df, target='Loan_Approved'):
    # Convert target to 0/1
    if df[target].dtype == 'O':
        df[target] = df[target].map({'No': 0, 'Yes': 1})
    return df

# Load data
df = load_data()
st.title("Credit Wise — Converted Notebook App")
st.markdown("This app converts the original notebook into an interactive Streamlit app with preprocessing, EDA, model training, and prediction.")

# Basic preview
st.subheader("Data preview")
st.dataframe(df.head(10))

# Preprocessing setup
df = preprocess_target(df)
num_cols, cat_cols = get_feature_lists(df)
st.sidebar.header("Preprocessing & Model")
st.sidebar.markdown(f"**Numeric cols:** {len(num_cols)} — **Categorical cols:** {len(cat_cols)}")

preprocessor = build_preprocessor(num_cols, cat_cols)

# EDA
st.subheader("Exploratory Data Analysis")

col1, col2 = st.columns([2,1])
with col1:
    st.markdown("**Loan approval distribution**")
    fig = px.histogram(df, x='Loan_Approved', color='Loan_Approved', labels={'Loan_Approved':'Loan Approved (0=No,1=Yes)'})
    st.plotly_chart(fig, use_container_width=True)

    if 'Applicant_Income' in df.columns:
        st.markdown("**Applicant Income distribution**")
        fig = px.histogram(df, x='Applicant_Income', nbins=50)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Missing values**")
    mv = df.isna().sum().sort_values(ascending=False)
    st.dataframe(mv[mv>0])

st.markdown("---")

# Train/Test split and model selection
X = df.drop(columns=['Loan_Approved'])
y = df['Loan_Approved']

st.sidebar.subheader("Train / Test Split")
test_size = st.sidebar.slider('Test size', min_value=0.1, max_value=0.5, value=0.2, step=0.05)
random_state = st.sidebar.number_input('Random state', value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

# Model selection
st.sidebar.subheader("Model")
model_name = st.sidebar.radio('Choose model', options=['Logistic Regression', 'K-Nearest Neighbors', 'Gaussian Naive Bayes'])

if model_name == 'K-Nearest Neighbors':
    n_neighbors = st.sidebar.slider('n_neighbors', min_value=1, max_value=25, value=5)
else:
    n_neighbors = None

train_button = st.sidebar.button('Train model')

# Train & evaluate
if train_button:
    if model_name == 'Logistic Regression':
        clf = LogisticRegression(max_iter=1000)
    elif model_name == 'K-Nearest Neighbors':
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        clf = GaussianNB()

    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('clf', clf)
    ])

    with st.spinner('Training...'):
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)[:, 1]

    st.subheader('Evaluation')
    st.write('Accuracy:', round(accuracy_score(y_test, y_pred), 3))
    st.write('Precision:', round(precision_score(y_test, y_pred, zero_division=0), 3))
    st.write('Recall:', round(recall_score(y_test, y_pred, zero_division=0), 3))
    st.write('F1:', round(f1_score(y_test, y_pred, zero_division=0), 3))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.subheader('Classification report')
    st.text(classification_report(y_test, y_pred, zero_division=0))

    # Save pipeline in session state for prediction UI
    st.session_state['pipeline'] = pipeline
else:
    st.info('Train a model using the sidebar to see evaluation metrics and enable prediction UI.')

st.markdown('---')

# Prediction UI
st.subheader('Make a prediction')
if 'pipeline' in st.session_state:
    pipeline = st.session_state['pipeline']

    st.markdown('Enter applicant details (leave defaults)')
    input_data = {}
    for c in X.columns:
        if c in num_cols:
            val = st.number_input(c, value=float(X[c].median()), format="%.2f")
            input_data[c] = float(val)
        else:
            opts = X[c].dropna().unique().tolist()
            default = opts[0] if opts else ''
            val = st.selectbox(c, options=opts, index=0)
            input_data[c] = val

    predict_btn = st.button('Predict')
    if predict_btn:
        row = pd.DataFrame([input_data])
        pred = pipeline.predict(row)[0]
        proba = None
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba(row)[0][1]

        st.write(f'Predicted: **{"Yes" if pred==1 else "No"}**')
        if proba is not None:
            st.write(f'Probability of approval: **{proba:.2f}**')
else:
    st.warning('No trained model available. Please train one from the sidebar.')

st.markdown('---')
st.caption('This app implements notebook preprocessing, EDA, model training, evaluation, and single-instance predictions. Adapt or extend as needed.')
