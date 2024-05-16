import streamlit as st
import tempfile
import sys
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Add your parent directory to the path
parent_dir = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base'
sys.path.append(parent_dir)
from Pipelines.Py_files.vision_pipeline_stlit import VisionPipeline
from TableReconstruction.image_processor import ImageProcessor
from TableReconstruction.text_classifier import TextClassifier
from TableReconstruction.row_clusterer_v2 import RowClassifier
from TableReconstruction.column_clusterer import ColumnClusterer
from TableReconstruction.streamlit_cluster_tuning import StreamlitClusterTuning


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
    gb.configure_column()
    # Configure column selection and interaction
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)

    # Build the grid options
    gridOptions = gb.build()

    # Enable fitting columns on grid load
    gridOptions['defaultColDef']['flex'] = 1

    # Setup the grid display options
    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        fit_columns_on_grid_load=True,  # Auto-fit column widths on load
        update_mode=GridUpdateMode.MODEL_CHANGED,  # Set the update mode for interaction
        key=key  # Use the key for maintaining session state
    )
    return grid_response

# def display_aggrid(df, key):
#     df.columns = [str(col) for col in df.columns]  # Ensure all column names are strings
#     gb = GridOptionsBuilder.from_dataframe(df)
#     gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
#     gb.configure_default_column(editable=True, resizable=True, groupable = True)  # Enable editable and resizable columns

#     gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)  # Enable column selection
#     gridOptions = gb.build()
#     row_height = 29  # Default row height
#     header_height = 56  # Default header height
#     total_rows = len(df)
#     grid_height = max(header_height + (total_rows * row_height), 100)  # Ensure a minimum height
#     grid_response = AgGrid(df, gridOptions=gridOptions, height=grid_height, fit_columns_on_grid_load=True, update_mode=GridUpdateMode.MODEL_CHANGED, key=key)


@st.cache_resource
def load_vision_pipeline(model_path):
    return VisionPipeline(model_path)
