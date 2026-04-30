import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap

from joblib import dump
from joblib import load


# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

file_path = os.path.join(project_root, 'Portfolio/X_train.csv')

dataset = pd.read_csv(file_path)
# Drop unnamed index columns if present
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]


# AWS Session Management
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)


# Data & Model Configuration
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "shap_explainer.pkl",
    "pipeline"  : "finalized_loan_model.tar.gz",
    "keys"      : ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_high'],
    "inputs"    : [
        {"name": "loan_amnt",       "type": "number", "min": 1000.0,  "max": 40000.0,  "default": 10000.0, "step": 500.0},
        {"name": "int_rate",        "type": "number", "min": 5.0,     "max": 30.0,     "default": 12.0,    "step": 0.5},
        {"name": "annual_inc",      "type": "number", "min": 10000.0, "max": 300000.0, "default": 60000.0, "step": 1000.0},
        {"name": "dti",             "type": "number", "min": 0.0,     "max": 50.0,     "default": 18.0,    "step": 0.5},
        {"name": "fico_range_high", "type": "number", "min": 600.0,   "max": 850.0,    "default": 700.0,   "step": 5.0}
    ]
}


def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}")

    # Extract the .joblib file from the .tar.gz
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    # Load the full pipeline
    return joblib.load(f"{joblib_file}")


def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return load(f)


# Prediction Logic
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df)
        # raw_pred is a list of predictions; take the first
        if isinstance(raw_pred, list):
            pred_val = int(raw_pred[0])
        else:
            pred_val = int(pd.DataFrame(raw_pred).values[-1][0])
        mapping = {0: "Fully Paid", 1: "Charged Off"}
        return mapping.get(pred_val, "Unknown"), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# Local Explainability
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # Build the preprocessing pipeline (everything except SMOTE + model)
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    input_df = pd.DataFrame(input_df)
    input_df_transformed = preprocessing_pipeline.transform(input_df)

    # Use the kbest selector to get feature names
    kbest_step = best_pipeline.named_steps['kbest']

    # Build manual feature names from the preprocessor
    preprocess_step = best_pipeline.named_steps['preprocess']
    manual_names = []
    for name, trans, cols in preprocess_step.transformers_:
        if name == 'cr_line':
            manual_names.append('earliest_cr_line_year')
        elif name == 'emp_len':
            manual_names.append('emp_length_num')
        elif name == 'term':
            manual_names.append('term_num')
        elif name == 'num':
            manual_names.extend(cols)
        elif name == 'cat':
            ohe = trans.named_steps['ohe']
            ohe_names = ohe.get_feature_names_out(cols)
            manual_names.extend(ohe_names)

    all_feature_names = np.array(manual_names)
    selected_mask = kbest_step.get_support()
    selected_features = all_feature_names[selected_mask]

    input_df_transformed = pd.DataFrame(input_df_transformed, columns=selected_features)
    shap_values = explainer(input_df_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))

    # Handle both 2D and 3D SHAP value formats
    if len(shap_values.shape) == 3:
        # Multi-class output: pick class 1 (Charged Off)
        shap.plots.waterfall(shap_values[0, :, 1])
        top_feature = pd.Series(
            shap_values[0, :, 1].values,
            index=shap_values[0, :, 1].feature_names
        ).abs().idxmax()
    else:
        shap.plots.waterfall(shap_values[0])
        top_feature = pd.Series(
            shap_values[0].values,
            index=shap_values[0].feature_names
        ).abs().idxmax()

    st.pyplot(fig)
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# Streamlit UI
st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("💰 Loan Default Predictor")
st.markdown("Enter applicant details below to predict the probability of loan default.")

with st.form("pred_form"):
    st.subheader("Applicant Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'],
                max_value=inp['max'],
                value=inp['default'],
                step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

# Build a complete input row by combining user inputs with defaults from X_train row 0
original = dataset.iloc[0:1].to_dict(orient='records')[0]
original.update(user_inputs)

if submitted:
    res, status = call_model_api([original])
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation([original], session, aws_bucket)
    else:
        st.error(res)
