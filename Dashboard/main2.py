import streamlit as st
from PIL import Image
import tempfile
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Add your parent directory to the path
parent_dir = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base'
sys.path.append(parent_dir)
from Pipelines.Py_files.vision_pipeline_stlit import VisionPipeline
from TableReconstruction.image_processor import ImageProcessor
from TableReconstruction.text_classifier import TextClassifier
from TableReconstruction.row_clusterer_v2 import RowClassifier
from TableReconstruction.column_clusterer import ColumnClusterer
from TableReconstruction.dataframe_processor import DataFrameProcessorV3
from TableReconstruction.streamlit_cluster_tuning import StreamlitClusterTuning

@st.experimental_singleton
def load_vision_pipeline():
    return VisionPipeline('/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/yolo_training/yolo_v8_models/finetune_v5/best.pt')

pipeline = load_vision_pipeline()

@st.experimental_memo(show_spinner=False)
def upload_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    return temp_file_path

@st.experimental_memo(show_spinner=False)
def predict_image(image_path, iou, conf, agnostic_nms):
    results = pipeline.predict(image_path, iou=iou, conf=conf, agnostic_nms=agnostic_nms)
    return results

@st.experimental_memo(show_spinner=False)
def process_image(image_path, boxes):
    image_processor = ImageProcessor(image_path, boxes)
    image = image_processor.image
    cropped_images = image_processor.cropped_images
    return image, cropped_images

@st.experimental_memo(show_spinner=False)
def classify_text(_cropped_images, _classes, _boxes):
    text_classifier = TextClassifier(_cropped_images, _classes, _boxes)
    return text_classifier

@st.experimental_memo(show_spinner=False)
def compute_best_threshold(_row_classifier, all_rows):
    best_threshold = _row_classifier.optimize_by_histogram(all_rows)
    return best_threshold

@st.experimental_memo(show_spinner=False)
def cluster_rows(_text_classifier):
    row_classifier = RowClassifier(_text_classifier.headers, _text_classifier.single_row, _text_classifier.tables)
    all_rows = row_classifier.collect_all_rows()
    return row_classifier, all_rows

@st.experimental_memo(show_spinner=False)
def cluster_columns(_regrouped_data):
    column_clusterer = ColumnClusterer(_regrouped_data)
    all_tables_data = column_clusterer.all_tables_data
    return column_clusterer, all_tables_data

@st.experimental_memo(show_spinner=False)
def process_tables_to_dataframe(tables_data, clusterer, min_samples_list):
    df_processor = DataFrameProcessorV3()
    final_dfs = df_processor.process_tables_to_dataframe(tables_data, clusterer, min_samples_list)
    return final_dfs

def display_aggrid(df, key):
    df.columns = [str(col) for col in df.columns]  # Ensure all column names are strings
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
    gb.configure_default_column(editable=True, resizable=True)  # Enable editable and resizable columns
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)  # Enable column selection
    gridOptions = gb.build()
    row_height = 25  # Default row height
    header_height = 56  # Default header height
    total_rows = len(df)
    grid_height = max(header_height + (total_rows * row_height), 100)  # Ensure a minimum height
    grid_response = AgGrid(df, gridOptions=gridOptions, height=grid_height, fit_columns_on_grid_load=True, update_mode=GridUpdateMode.MODEL_CHANGED, key=key)
    selected = grid_response.get('selected_rows', [])
    selected_columns = [col['field'] for col in selected] if selected else []
    return selected_columns

def display_image_with_boxes(image_path, results):
    result = results[0]
    boxes = result.boxes.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    img_with_boxes_path = pipeline.visualize_boxes(image_path, boxes, classes, pipeline.class_names)
    img_with_boxes = Image.open(img_with_boxes_path)
    st.image(img_with_boxes, caption="Image with Bounding Boxes", use_column_width=True)
    return boxes, classes

def merge_columns(df, columns_to_merge):
    if columns_to_merge:
        df['Merged'] = df[columns_to_merge].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        df.drop(columns=columns_to_merge, inplace=True)
    return df

def main():
    st.title("Information Extraction - Table Reconstruction Pipeline")

    # Step 1: Upload Image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        # Clear cache when a new file is uploaded
        if "uploaded_file" in st.session_state and st.session_state.uploaded_file != uploaded_file:
            st.experimental_memo.clear()

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
        text_classifier = classify_text(cropped_images, classes, boxes)

        # Step 6: Cluster the Rows
        row_classifier, all_rows = cluster_rows(text_classifier)
        st.write(f"Number of rows for clustering: {len(all_rows)}")

        # Step 7: Interactive Cluster Tuning
        st.subheader("Interactive Row Cluster Tuning")
        if 'cluster_tuning' not in st.session_state:
            st.session_state.cluster_tuning = StreamlitClusterTuning(image, row_classifier, all_rows)
            st.session_state.best_threshold = compute_best_threshold(row_classifier, all_rows)
        
        best_threshold = st.slider("Threshold for Clustering", min_value=0.1, max_value=0.8, value=st.session_state.best_threshold, step=0.001)
        st.session_state.cluster_tuning.update_clustering(best_threshold)

        # Step 8: Cluster Columns
        regrouped_data = st.session_state.cluster_tuning.grouped_data
        column_clusterer, all_tables_data = cluster_columns(regrouped_data)

        # Step 9: Process Tables into DataFrames
        st.subheader("Clustering Strength for Each Table:")
        min_samples_list = []
        for idx, table_data in enumerate(all_tables_data):
            min_samples = st.slider(f"Clustering Strength for Table {idx + 1}", min_value=1, max_value=10, value=4, step=1)
            min_samples_list.append(min_samples)

        final_dfs = process_tables_to_dataframe(all_tables_data, column_clusterer, min_samples_list)

        # Initialize session state for DataFrame and selected columns
        if 'dataframes' not in st.session_state:
            st.session_state.dataframes = final_dfs
        if 'selected_columns' not in st.session_state:
            st.session_state.selected_columns = {}

        # Display Results
        st.subheader("Extracted Dataframes:")
        st.write("Please select grade table for analysis.")
        for idx, df in enumerate(st.session_state.dataframes):
            st.write(f"\nTable {idx + 1} Preview:\n")
            selected_columns = display_aggrid(df, key=f"table_{idx}")

            if selected_columns:
                st.session_state.selected_columns[f"table_{idx}"] = selected_columns
                if st.button(f"Merge selected columns for Table {idx + 1}", key=f"merge_button_{idx}"):
                    df = merge_columns(df, selected_columns)
                    st.session_state.dataframes[idx] = df
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
