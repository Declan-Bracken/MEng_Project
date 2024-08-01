import streamlit as st
import tempfile
import sys
# import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import os

current_directory = os.getcwd()
print("Current working directory:", current_directory)
items = os.listdir(current_directory)

folders = [item for item in items if os.path.isdir(os.path.join(current_directory, item))]

print("Folders in the current directory:", folders)

# Add your parent directory to the path
parent_dir = r'/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base'
sys.path.append(parent_dir)
from Pipelines.Py_files.vision_pipeline_stlit import VisionPipeline
from TableReconstruction.image_processor import ImageProcessor
from TableReconstruction.text_classifier import TextClassifier
from TableReconstruction.row_clusterer_v2 import RowClassifier
from TableReconstruction.column_clusterer import ColumnClusterer
from Dashboard.Transcript_Processor.utils.streamlit_cluster_tuning import StreamlitClusterTuning

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

# Function to adjust DataFrame column widths based on content
# def calculate_column_widths(df):
#     max_lengths = {}
#     for col in df.columns:
#         max_length = df[col].astype(str).map(len).max()
#         max_lengths[col] = max_length
#     return max_lengths

# def generate_column_config(df):
#     col_widths = calculate_column_widths(df)
#     column_config = {}
#     for col, width in col_widths.items():
#         column_config[col] = st.column_config.TextColumn(
#             width=f"{width + 5}em"  # Adjust width with padding
#         )
#     return column_config

# def display_adjusted_dataframe(df):
#     col_config = generate_column_config(df)
#     st.experimental_dataframe(df, column_config=col_config)


# def display_aggrid(df, key):
#     df.columns = [str(col) for col in df.columns]  # Ensure all column names are strings
#     gb = GridOptionsBuilder.from_dataframe(df)
#     gb.configure_grid_options(ColumnsAutoSizeMode = 'fitGridWidth')
#     gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
#     gb.configure_default_column(editable=True, resizable=True, groupable = True)  # Enable editable and resizable columns
#     column_defs = gridOptions["columnDefs"]
#         for col_def in column_defs:
#             col_name = col_def["field"]
#             max_len = df[col_name].astype(str).str.len().max() # can add +5 here if things are too tight
#             col_def["width"] = max_len
#     # gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)  # Enable column selection
#     gridOptions = gb.build()
#     # row_height = 29  # Default row height
#     # header_height = 56  # Default header height
#     # total_rows = len(df)
#     # grid_height = max(header_height + (total_rows * row_height), 100)  # Ensure a minimum height
#     #gridOptions=gridOptions, height=grid_height
#     grid_response = AgGrid(df, fit_columns_on_grid_load=True,gridOptions=gridOptions,update_mode=GridUpdateMode.MODEL_CHANGED, key=key, )


@st.cache_resource
def load_vision_pipeline(model_path):
    return VisionPipeline(model_path)
