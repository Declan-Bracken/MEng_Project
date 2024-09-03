import streamlit as st
from PIL import Image
import tempfile
import sys
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Add your parent directory to the path
parent_dir = r'c:\Users\Declan Bracken\MEng_Project'
sys.path.append(parent_dir)
from Pipelines.Py_files.vision_pipeline_stlit import VisionPipeline
from TableReconstruction.image_processor import ImageProcessor
from TableReconstruction.text_classifier import TextClassifier
from TableReconstruction.row_clustering.row_clusterer_v2 import RowClassifier
from TableReconstruction.column_clustering.column_clusterer import ColumnClusterer
from TableReconstruction.streamlit_cluster_tuning import StreamlitClusterTuning


# def set_default_model_path(new_path):
#     st.session_state.default_model_path = new_path

@st.cache_data(show_spinner=False)
def upload_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    return temp_file_path

@st.cache_data(show_spinner=False)
def predict_image(image_path, iou, conf, agnostic_nms):
    return pipeline.predict(image_path, iou=iou, conf=conf, agnostic_nms=agnostic_nms)

@st.cache_data(show_spinner=False)
def display_image_with_boxes(image_path, _results):
    result = _results[0]
    boxes = result.boxes.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    img_with_boxes_path = pipeline.visualize_boxes(image_path, boxes, classes, pipeline.class_names)
    img_with_boxes = Image.open(img_with_boxes_path)
    st.image(img_with_boxes, caption="Image with Bounding Boxes", use_column_width=True)
    return boxes, classes

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
def initialize_cluster_tuning(image, _row_classifier, all_rows):
    return StreamlitClusterTuning(image, _row_classifier, all_rows)

# @st.cache_data(show_spinner=False)
def compute_best_threshold(row_classifier, all_rows):
    best_threshold = row_classifier.optimize_by_histogram(all_rows)
    return best_threshold

# Cached Resource
@st.cache_resource(show_spinner=False)
def load_cluster_rows(headers, single_row, tables):
    return RowClassifier(headers, single_row, tables)

# Resource
@st.cache_resource(show_spinner=False)
def load_column_clusterer(grouped_data):
    return ColumnClusterer(grouped_data)

def display_aggrid(df, key):
    df.columns = [str(col) for col in df.columns]  # Ensure all column names are strings
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
    gb.configure_default_column(editable=True, resizable=True, groupable = True)  # Enable editable and resizable columns

    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)  # Enable column selection
    gridOptions = gb.build()
    row_height = 29  # Default row height
    header_height = 56  # Default header height
    total_rows = len(df)
    grid_height = max(header_height + (total_rows * row_height), 100)  # Ensure a minimum height
    grid_response = AgGrid(df, gridOptions=gridOptions, height=grid_height, fit_columns_on_grid_load=True, update_mode=GridUpdateMode.MODEL_CHANGED, key=key)


@st.cache_resource
def load_vision_pipeline(model_path = r'C:\Users\Declan Bracken\MEng_Project\yolo_training\yolo_v8_models\finetune_v4 (3_classes)\best (1).pt'):
    return VisionPipeline(model_path)

def main():
    global pipeline
    st.title("Information Extraction - Table Reconstruction Pipeline")
    # Load Vision Pipeline
    pipeline = load_vision_pipeline()
    
    # Step 1.5: Upload Image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        # Clear cache when a new file is uploaded
        if "uploaded_file" in st.session_state and st.session_state.uploaded_file != uploaded_file:
            st.cache_data.clear()

        st.session_state.uploaded_file = uploaded_file
        image_path = upload_image(uploaded_file)

        # Step 2 & 3: Run YOLO and Display Image with Bounding Boxes
        st.subheader("YOLO Model Parameters")
        iou = st.slider("IoU Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        conf = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        agnostic_nms = st.checkbox("Agnostic NMS", value=True)

        results = predict_image(image_path, iou, conf, agnostic_nms)
        boxes, classes = display_image_with_boxes(image_path, results)

        # Step 4: Process the Image
        image, cropped_images = process_image(image_path, boxes)
        st.write(f"Number of cropped images: {len(cropped_images)}")

        # Step 5: Classify the Text
        text_classifier = classify_text(cropped_images, classes, boxes) # Create Resource
        headers = text_classifier.headers
        single_row = text_classifier.single_row
        tables = text_classifier.tables

        # Step 6: Cluster the Rows
        row_classifier = load_cluster_rows(headers, single_row, tables)
        all_rows = row_classifier.collect_all_rows()
        st.write(f"Number of rows for clustering: {len(all_rows)}")

        # Step 7: Interactive Cluster Tuning
        st.subheader("Interactive Row Cluster Tuning")
        cluster_tuning = initialize_cluster_tuning(image, row_classifier, all_rows)
        best_threshold = compute_best_threshold(row_classifier, all_rows)
        best_threshold = st.slider("Threshold for Clustering", min_value=0.1, max_value=0.8, value=best_threshold, step=0.001)
        cluster_tuning.update_clustering(best_threshold)

        # Step 8: Cluster Columns
        grouped_data = cluster_tuning.grouped_data  # This would be obtained from the GUI in the original code
        column_clusterer = load_column_clusterer(grouped_data)

        # Step 9: Process Tables into DataFrames
        min_samples_list = []
        for idx in range(len(grouped_data)):
            min_samples = st.slider(f"Clustering Strength for Table {idx + 1}", min_value=1, max_value=10, value=4, step=1)
            min_samples_list.append(min_samples)

        final_dfs = column_clusterer.process_tables_to_dataframe(grouped_data, min_samples_list)
        
        # Initialize session state for DataFrame and selected columns
        if 'dataframes' not in st.session_state:
            st.session_state.dataframes = final_dfs

        # Display Results
        st.subheader("Extracted Dataframes:")
        st.write("Please select grade table for analysis.")
        for idx, df in enumerate(final_dfs):
            st.write(f"\nTable {idx + 1} Preview:\n")
            display_aggrid(df, key=f"table_{idx}")

if __name__ == "__main__":
    main()
