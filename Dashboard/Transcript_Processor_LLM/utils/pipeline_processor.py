import streamlit as st
import tempfile
import sys
import torch
# import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Add your parent directory to the path
parent_dir = r'c:\Users\Declan Bracken\MEng_Project'
sys.path.append(parent_dir)
from Pipelines.Py_files.vision_pipeline_stlit import VisionPipeline
from TableReconstruction.image_processor import ImageProcessor
from TableReconstruction.text_classifier import TextClassifier
from TableReconstruction.row_clusterer_v2 import RowClassifier
from TableReconstruction.column_clusterer import ColumnClusterer
from Dashboard.Transcript_Processor.utils.streamlit_cluster_tuning import StreamlitClusterTuning
# For LLM Inference
from Pipelines.Py_files.mistral_pipeline_v2 import MistralInference

def set_default_model_path(new_path):
    st.session_state.default_model_path = new_path

@st.cache_data(show_spinner=False)
def upload_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    return temp_file_path

@st.cache_data(show_spinner=False)
def process_image(image_path, boxes):
    image_processor = ImageProcessor(image_path, boxes)
    image = image_processor.image
    cropped_images = image_processor.cropped_images
    return image, cropped_images

# Cached Resource
@st.cache_resource(show_spinner=False)
def classify_text(cropped_images, classes, boxes):
    return TextClassifier(cropped_images, classes, boxes)

# Cached Resource
@st.cache_resource(show_spinner=False)
def load_cluster_rows(headers, single_row, tables):
    return RowClassifier(headers, single_row, tables)

# Cached Resource
@st.cache_resource(show_spinner=False)
def initialize_cluster_tuning(image, _row_classifier, all_rows):
    return StreamlitClusterTuning(image, _row_classifier, all_rows)

@st.cache_data(show_spinner=False)
def compute_best_threshold(_row_classifier, all_rows):
    best_threshold = _row_classifier.optimize_by_histogram(all_rows)
    return best_threshold

# Resource
@st.cache_resource(show_spinner=False)
def load_column_clusterer(grouped_data):
    return ColumnClusterer(grouped_data)

def display_aggrid(df, key):
    df.columns = [str(col) for col in df.columns]  # Ensure all column names are strings

    # Initialize GridOptionsBuilder with the dataframe
    gb = GridOptionsBuilder.from_dataframe(df)
    # Configure pagination to automatically adjust the page size to fit the grid height
    gb.configure_pagination(paginationAutoPageSize=True)
    # Configure default column settings to allow for edits, resizing, and grouping
    gb.configure_default_column(editable=True, resizable=True, autoHeight=True, wrapText=True)
    # gb.configure_side_bar(columns_panel = True,defaultToolPanel='columns')
    # Configure column selection and interaction
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)

    # Build the grid options
    gridOptions = gb.build()

    column_defs = gridOptions["columnDefs"]
    for col_def in column_defs:
        col_name = col_def["field"]
        max_len = df[col_name].astype(str).str.len().max() # can add +5 here if things are too tight
        col_def["width"] = max_len

    # Enable fitting columns on grid load
    # gridOptions['defaultColDef']['flex'] = 1

    # Setup the grid display options
    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        fit_columns_on_grid_load=True,  # Auto-fit column widths on load
        update_mode=GridUpdateMode.MODEL_CHANGED,  # Set the update mode for interaction
        key=key  # Use the key for maintaining session state
    )
    return grid_response

@st.cache_resource
def load_vision_pipeline(model_path):
    return VisionPipeline(model_path)

@st.cache_resource(show_spinner=False)
def load_mistral(model_path = r"C:\Users\Declan Bracken\MEng_Project\mistral\models\dolphin-2.1-mistral-7b.Q5_K_M.gguf"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mistral_pipeline = MistralInference(device=device, model_path=model_path)
    return mistral_pipeline

@st.cache_data(show_spinner=False)
def query_mistral(headers, selected_table_text, mistral_model_path):
    st.write("Querying Mistral...")
    mistral_pipeline = load_mistral(mistral_model_path)
    if headers:
        final_df = mistral_pipeline.process_transcript(headers, selected_table_text)
    else:
        final_df = mistral_pipeline.process_transcript(None, selected_table_text)
    
    return final_df
    
    # # Store the resulting dataframe in session state
    # st.session_state.final_df = final_df

@st.cache_data
def groups_to_strings(grouped_data):
    strings = []
    for group in grouped_data:
        lines = {}
        for line_num, text in zip(group["line_numbers"], group["texts"]):
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(text)
        
        ordered_lines = [f"{' '.join(words)}" for _, words in sorted(lines.items())]
        strings.append('\n'.join(ordered_lines))
    return strings

@st.cache_data
def headers_to_strings(header_lines):
    strings = []
    for headers in header_lines:
        ordered_lines = headers["text"]["text"]
        strings.append(' '.join(ordered_lines).strip())
    return strings

# Function to save DataFrame to selected directory
def save_dataframe(df, directory):
    if directory:
        file_path = f"{directory}/grade_dataframe.csv"
        df.to_csv(file_path, index=False)
        st.success(f"DataFrame saved successfully at {file_path}!")
    else:
        st.error("No directory selected. Please select a directory.")