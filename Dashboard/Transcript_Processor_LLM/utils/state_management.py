import streamlit as st
import pandas as pd

def initialize_state():
    # For main
    # Initialize session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Main"
    
    # For Extraction
    if 'vision_model_path' not in st.session_state:
        st.session_state.vision_model_path = r'C:\Users\Declan Bracken\MEng_Project\yolo_training\yolo_v8_models\finetune_v5\best.pt' #r'C:\Users\Declan Bracken\MEng_Project\yolo_training\yolo_v8_models\finetune_v5\best.pt' # Default vision path
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'vision_model_loaded' not in st.session_state:
        st.session_state.vision_model_loaded = False
    if 'grouped_data' not in st.session_state:
        st.session_state.grouped_data = None
    if 'string_nlines' not in st.session_state:
        st.session_state.string_nlines = None
    if 'df_nlines' not in st.session_state:
        st.session_state.df_nlines = None
    if 'llm_path' not in st.session_state:
        st.session_state.llm_path = r'mistral\models\laser-dolphin-mixtral-2x7b-dpo.Q5_K_M.gguf' #r'C:\Users\Declan Bracken\MEng_Project\mistral\models\laser-dolphin-mixtral-2x7b-dpo.Q5_K_M.gguf'
    if 'final_df' not in st.session_state:
        st.session_state.final_df = None

    # For Analytics
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = pd.DataFrame()
    if 'result_df' not in st.session_state:
        st.session_state.result_df = pd.DataFrame()

def update_state(key, value):
    st.session_state[key] = value
