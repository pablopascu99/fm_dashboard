import streamlit as st
import pandas as pd
from utils import standardize_column_names
from tabs import render_metrics_dashboard, render_anomaly_detection, column_mapping

# App configuration
st.set_page_config(
    layout="wide",
    page_title="Network Metrics Dashboard",
    page_icon="📡",
    initial_sidebar_state="expanded"
)

# Load the CSV file
df = pd.read_csv("src/files/all_data_ts.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = standardize_column_names(df, column_mapping)

# Create tabs
tabs = st.tabs(["Metrics Dashboard", "Anomaly Detection"])
with tabs[0]:
    render_metrics_dashboard(df)
with tabs[1]:
    render_anomaly_detection()