import streamlit as st

def initialize_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'Extraction'
    if 'model_path' not in st.session_state:
        st.session_state.model_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/yolo_training/yolo_v8_models/finetune_v5/best.pt'
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'vision_model_loaded' not in st.session_state:
        st.session_state.vision_model_loaded = False
    if 'grouped_data' not in st.session_state:
        st.session_state.grouped_data = None
    if 'final_dfs' not in st.session_state:
        st.session_state.final_dfs = None

def update_state(key, value):
    st.session_state[key] = value
