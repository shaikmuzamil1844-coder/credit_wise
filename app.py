import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import glob
import joblib

st.set_page_config(page_title="Credit Wise - Loan Explorer & Model", layout="wide")

@st.cache_data
def load_data(path="loan_approval_data.csv"):
    df = pd.read_csv(path)
    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data
def preprocess_target(df, target='Loan_Approved'):
    if df[target].dtype == 'O':
        df[target] = df[target].map({'No': 0, 'Yes': 1})
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

# Load and prepare data
raw_df = load_data()
raw_df = preprocess_target(raw_df)

# --- Add a polished dark theme header / hero and KPIs ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    .stApp { background: linear-gradient(180deg, #061426 0%, #07122a 50%, #0b0f1a 100%); color: #e6eef8; font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; }
    .hero { padding: 20px; border-radius: 10px; background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow: 0 10px 30px rgba(0,0,0,0.6); display:flex; align-items:center; gap:18px; }
    .hero .logo { width:56px; height:56px; border-radius:10px; background: linear-gradient(45deg,#06bf9a,#0b69a3); display:flex; align-items:center; justify-content:center; color:#031a11; font-weight:700; font-size:20px; }
    .hero h1 { margin: 0; color: #ffffff; font-size: 28px; }
    .hero p { margin: 6px 0 0; color: #a8b6cc; }
    .card { background: rgba(255,255,255,0.03); border-radius: 8px; padding: 14px; border: 1px solid rgba(255,255,255,0.06); box-shadow: 0 6px 18px rgba(0,0,0,0.6); color: #e6eef8; backdrop-filter: blur(6px); }
    .metric-title { color: #9fb3d3; font-weight: 600; font-size:13px; }
    .sidebar .stButton>button { background-color:#06bf9a; color:#031a11; }
    </style>
    """,
    unsafe_allow_html=True,
)

# set plotly look and color palette for dark theme
px.defaults.template = "plotly_dark"
COLOR_PALETTE = ['#00b894','#0984e3','#fdcb6e','#e17055','#6c5ce7']
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Header style controls
header_style = st.sidebar.selectbox('Header style', options=['Modern', 'Elegant', 'Classic'], index=0)
header_size = st.sidebar.slider('Header size (px)', min_value=24, max_value=48, value=32)

# Build header HTML with inline styles based on selection
if header_style == 'Modern':
    h_style = f"font-family: 'Segoe UI', Roboto, Arial, sans-serif; font-weight:700; font-size:{header_size}px; letter-spacing:1px; background:linear-gradient(90deg,#06bf9a,#0b69a3); -webkit-background-clip:text; color:transparent; text-shadow: 0 2px 10px rgba(6, 191, 154, 0.12);"
elif header_style == 'Elegant':
    h_style = f"font-family: Georgia, 'Times New Roman', serif; font-style:italic; font-weight:700; font-size:{header_size}px; color:#ffffff; text-shadow: 0 2px 8px rgba(0,0,0,0.6);"
else:
    h_style = f"font-family: 'Segoe UI', Roboto, Arial, sans-serif; font-variant:small-caps; letter-spacing:2px; font-size:{header_size}px; color:#ffffff; text-shadow: 0 1px 2px rgba(0,0,0,0.6);"

subtitle_style = "color:#a8b6cc; margin:6px 0 0;"

header_html = f"<div class='hero'><div class='logo'>CW</div><div><h1 style=\"{h_style}\">Credit Wise</h1><p style=\"{subtitle_style}\">Interactive loan approval explorer — analyze data, train models, and predict decisions with an advanced dark UI.</p></div></div>"

st.markdown(header_html, unsafe_allow_html=True)

# KPIs
total = len(raw_df)
approved_rate = raw_df['Loan_Approved'].mean() * 100
avg_loan = raw_df['Loan_Amount'].median()
avg_credit = raw_df['Credit_Score'].median()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='card'><div class='metric-title'>Total Applicants</div><div style='font-size:20px;font-weight:700'>{total}</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='card'><div class='metric-title'>Approval Rate</div><div style='font-size:20px;font-weight:700'>{approved_rate:.1f}%</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='card'><div class='metric-title'>Median Loan</div><div style='font-size:20px;font-weight:700'>₹{avg_loan:,.0f}</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='card'><div class='metric-title'>Median Credit Score</div><div style='font-size:20px;font-weight:700'>{avg_credit:.0f}</div></div>", unsafe_allow_html=True)

st.markdown("---")

st.markdown("Explore the dataset or train a model to predict loan approval. Use the sidebar to switch pages.")

page = st.sidebar.radio("Page", ["Explorer", "Model"])

# Explorer page
if page == "Explorer":
    st.sidebar.header("Filters")
    df = raw_df.copy()

    employment = st.sidebar.multiselect("Employment Status", options=sorted(df['Employment_Status'].dropna().unique()), default=sorted(df['Employment_Status'].dropna().unique()))
    area = st.sidebar.multiselect("Property Area", options=sorted(df['Property_Area'].dropna().unique()), default=sorted(df['Property_Area'].dropna().unique()))
    purpose = st.sidebar.multiselect("Loan Purpose", options=sorted(df['Loan_Purpose'].dropna().unique()), default=sorted(df['Loan_Purpose'].dropna().unique()))
    approved = st.sidebar.multiselect("Loan Approved", options=sorted(df['Loan_Approved'].dropna().unique()), default=sorted(df['Loan_Approved'].dropna().unique()))

    # Numeric range filters
    min_credit, max_credit = int(df['Credit_Score'].min(skipna=True)), int(df['Credit_Score'].max(skipna=True))
    credit_range = st.sidebar.slider('Credit Score range', min_value=min_credit, max_value=max_credit, value=(min_credit, max_credit))

    # Apply filters
    mask = df['Employment_Status'].isin(employment) & df['Property_Area'].isin(area) & df['Loan_Purpose'].isin(purpose) & df['Loan_Approved'].isin(approved)
    mask = mask & df['Credit_Score'].between(credit_range[0], credit_range[1])
    filtered = df[mask]

    st.markdown(f"**Rows:** {len(filtered)} (filtered) — **Columns:** {len(df.columns)}")

    # Top row: data preview and summary
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Data preview")
        st.dataframe(filtered.head(25), use_container_width=True)
        st.download_button("Download filtered CSV", data=filtered.to_csv(index=False), file_name="loan_filtered.csv")

    with col2:
        st.subheader("Summary stats")
        st.dataframe(filtered.select_dtypes('number').describe().round(2))
        st.write("Missing values (by colu@mn):")
        mv = filtered.isna().sum().sort_values(ascending=False)
        st.dataframe(mv[mv>0])

    st.markdown("---")

    # Visualizations
    st.subheader("Key Charts")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Loan approval counts**")
        fig = px.histogram(filtered, x='Loan_Approved', color='Loan_Approved', barnorm=None, title='Loan Approved vs Not', color_discrete_sequence=COLOR_PALETTE)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Applicant Income distribution**")
        fig = px.histogram(filtered, x='Applicant_Income', nbins=50, marginal='box', title='Applicant Income', color_discrete_sequence=COLOR_PALETTE)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Loan Amount distribution**")
        fig = px.histogram(filtered, x='Loan_Amount', nbins=50, marginal='box', title='Loan Amount', color_discrete_sequence=COLOR_PALETTE)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Credit Score vs Loan Amount (scatter)**")
        fig = px.scatter(filtered, x='Credit_Score', y='Loan_Amount', color='Loan_Approved', hover_data=['Applicant_Income','Loan_Purpose'], title='Credit Score vs Loan Amount', color_discrete_sequence=COLOR_PALETTE)
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.markdown("**Loan Amount by Employment Status (box)**")
        fig = px.box(filtered, x='Employment_Status', y='Loan_Amount', color='Loan_Approved', points='outliers', title='Loan Amount by Employment', color_discrete_sequence=COLOR_PALETTE)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Numeric correlation")
    num = filtered.select_dtypes('number').drop(columns=['Applicant_ID'], errors='ignore')
    if num.shape[1] >= 2:
        corr = num.corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
        fig.update_layout(height=500, title='Correlation matrix')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Not enough numeric columns to show correlation matrix.")

# Model page
else:
    st.sidebar.header("Model & Training")
    st.subheader("Modeling: Train and predict loan approval")

    df = raw_df.copy()
    if 'Applicant_ID' in df.columns:
        df = df.drop(columns=['Applicant_ID'])

    num_cols, cat_cols = get_feature_lists(df)

    st.markdown("**Data preview**")
    st.dataframe(df.head(10))

    st.markdown("---")

    st.sidebar.subheader("Train / Test")
    test_size = st.sidebar.slider('Test size', min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.sidebar.number_input('Random state', value=42, step=1)

    st.sidebar.subheader("Model selection")
    model_name = st.sidebar.radio('Model', options=['Logistic Regression', 'K-Nearest Neighbors', 'GaussianNB'])
    if model_name == 'K-Nearest Neighbors':
        n_neighbors = st.sidebar.slider('n_neighbors', min_value=1, max_value=25, value=5)
    else:
        n_neighbors = None

    train_btn = st.sidebar.button('Train model')

    # Train
    if train_btn:
        preproc = build_preprocessor(num_cols, cat_cols)
        X = df.drop(columns=['Loan_Approved'])
        y = df['Loan_Approved']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

        if model_name == 'Logistic Regression':
            clf = LogisticRegression(max_iter=1000)
        elif model_name == 'K-Nearest Neighbors':
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            clf = GaussianNB()

        pipeline = Pipeline([
            ('preproc', preproc),
            ('clf', clf)
        ])

        with st.spinner('Training model...'):
            pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        st.subheader('Evaluation')
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Accuracy', f"{acc:.3f}")
        m2.metric('Precision', f"{prec:.3f}")
        m3.metric('Recall', f"{rec:.3f}")
        m4.metric('F1', f"{f1:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        st.subheader('Classification report')
        st.text(classification_report(y_test, y_pred, zero_division=0))

        st.success('Model trained — available for predictions on this page.')
        st.session_state['pipeline'] = pipeline

    else:
        st.info('Train a model using the sidebar controls to enable prediction UI.')

    st.markdown('---')

    st.subheader('Saved models')
    files = sorted(glob.glob(os.path.join(MODEL_DIR, '*.joblib')), reverse=True)
    if files:
        display_files = [os.path.basename(f) for f in files]
        sel = st.selectbox('Choose a saved model', options=display_files)
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button('Load selected model'):
                pipeline_loaded = joblib.load(os.path.join(MODEL_DIR, sel))
                st.session_state['pipeline'] = pipeline_loaded
                st.success(f'Loaded {sel}')
        with col2:
            if st.button('Delete selected model'):
                os.remove(os.path.join(MODEL_DIR, sel))
                st.success(f'Deleted {sel}')
                st.experimental_rerun()
        with col3:
            if st.button('Download selected model'):
                with open(os.path.join(MODEL_DIR, sel), 'rb') as f:
                    st.download_button('Download model', data=f, file_name=sel)
    else:
        st.info('No saved models found. Save one after training.')

    st.markdown('---')
    st.subheader('Prediction')

    if 'pipeline' in st.session_state:
        pipeline = st.session_state['pipeline']
        st.markdown('Enter applicant details:')
        input_data = {}
        X_cols = df.drop(columns=['Loan_Approved']).columns
        for c in X_cols:
            if c in num_cols:
                val = st.number_input(c, value=float(df[c].median()), format="%.2f")
                input_data[c] = float(val)
            else:
                opts = df[c].dropna().unique().tolist()
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

st.sidebar.markdown("---")
st.sidebar.write("Built with Streamlit — run with `streamlit run app.py`")

# Footer notes
st.caption("Tip: switch between Explorer and Model pages using the sidebar, and train models to try predictions.")
