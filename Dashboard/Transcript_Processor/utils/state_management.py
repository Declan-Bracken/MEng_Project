import streamlit as st
import pandas as pd

def initialize_state():
    # For main
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Main"
    
    # For Extraction
    if 'model_path' not in st.session_state:
        st.session_state.model_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/yolo_training/yolo_v8_models/finetune_v5/best.pt'
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'vision_model_loaded' not in st.session_state:
        st.session_state.vision_model_loaded = False
    if 'grouped_data' not in st.session_state:
        st.session_state.grouped_data = None
    if 'min_samples_list' not in st.session_state:
        st.session_state.min_samples_list = []
    if 'cluster_selection_epsilon' not in st.session_state:
        st.session_state.cluster_selection_epsilon = 0.00  # Default value
    if 'final_dfs' not in st.session_state:
        st.session_state.final_dfs = None

    # For Analytics
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = pd.DataFrame()
    if 'selected_df_index' not in st.session_state:
        st.session_state.selected_df_index = None
    if 'result_df' not in st.session_state:
        st.session_state.result_df = pd.DataFrame()

def update_state(key, value):
    st.session_state[key] = value
