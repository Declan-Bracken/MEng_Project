import streamlit as st
from PIL import Image
from utils.state_management import initialize_state, update_state
import utils.pipeline_processor as PP
import numpy as np
import os
import io

@st.cache_data(show_spinner=False)
def predict_image(image_path, iou, conf, agnostic_nms):
    return vision_pipeline.predict(image_path, iou=iou, conf=conf, agnostic_nms=agnostic_nms)

@st.cache_data(show_spinner=False)
def display_image_with_boxes(image_path, boxes, classes):
    img_with_boxes_path = vision_pipeline.visualize_boxes(image_path, boxes, classes, vision_pipeline.class_names)
    img_with_boxes = Image.open(img_with_boxes_path)
    return img_with_boxes

# Example image directory
EXAMPLE_IMAGE_DIR = "Dashboard/Transcript_Processor/assets"

# Example images
example_images = {
    "Example 1": os.path.join(EXAMPLE_IMAGE_DIR, "2015-queens-university-transcript-1-2048.webp"),
    "Example 2": os.path.join(EXAMPLE_IMAGE_DIR, "unofficial-undergraduate-transcript-1-2048.webp"),
    "Example 3": os.path.join(EXAMPLE_IMAGE_DIR, "transcript-1-2-2048.webp"),
}

# @st.experimental_fragment
def app(): 
    # update_state("page", "Extraction")
    global vision_pipeline
    st.title("Information Extraction - Table Reconstruction Pipeline")
    
    # Initialize session state
    initialize_state()
    
    # Step 1.5: Upload Image
    # Option to select an example image
    example_option = st.radio("Select an example image", list(example_images.keys()), index=0)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], key='file_uploader')

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button("Process Image"):
            update_state('uploaded_file', uploaded_file)
    else:
        example_image_path = example_images[example_option]
        with open(example_image_path, 'rb') as f:
            example_image = f.read()
        example_file = io.BytesIO(example_image)
        example_file.name = example_image_path  # Set the name attribute
        image = Image.open(example_file)
        st.image(image, caption=f'Selected Example Image: {example_option}', use_column_width=True)
        if st.button("Process Image"):
            update_state('uploaded_file', example_file)

    if st.session_state.uploaded_file is not None:

        # Clear cache and session state when a new file is uploaded
        if st.session_state.uploaded_file != uploaded_file:
            st.cache_data.clear()

        #----- CACHED -----
        image_path = PP.upload_image(st.session_state.uploaded_file)

        # Step 1: Set Vision Model Path
        st.subheader("Set Vision Model Path")
        model_path = st.text_input("Vision Model Path", value=st.session_state.model_path)
        if st.button("Set Model Path"):
            update_state('model_path', model_path) # Update status of vision model path
            update_state('vision_model_loaded', False) # Update status of vision model
            st.success("Changed Model Path!")
            
        # Load Vision Pipeline
        #----- CACHED -----
        vision_pipeline = PP.load_vision_pipeline(model_path)
        update_state('vision_model_loaded', True) # Update status of vision model

        # Step 2 & 3: Run YOLO and Display Image with Bounding Boxes
        st.subheader("YOLO Model Parameters")
        iou = st.slider("IoU Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        conf = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        agnostic_nms = st.checkbox("Agnostic NMS", value=True)

        #----- CACHED -----
        results = predict_image(image_path, iou, conf, agnostic_nms)

        #----- NOT CACHED -----
        result = results[0]
        boxes = result.boxes.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        #----- CACHED -----
        img_with_boxes = display_image_with_boxes(image_path, boxes, classes)
        # display image 
        #----- NOT CACHED -----
        st.image(img_with_boxes, caption="Image with Bounding Boxes", use_column_width=True)

        # Step 4: Process the Image
        #----- CACHED -----
        image, cropped_images = PP.process_image(image_path, boxes)
        st.write(f"Number of cropped images: {len(cropped_images)}")

        # Step 5: Classify the Text
        #----- CACHED -----
        text_classifier = PP.classify_text(cropped_images, classes, boxes)

        #----- NOT CACHED -----
        headers = text_classifier.headers
        single_row = text_classifier.single_row
        tables = text_classifier.tables

        # Step 6: Cluster the Rows
        #----- CACHED -----
        row_classifier = PP.load_cluster_rows(headers, single_row, tables)
        #----- NOT CACHED -----
        all_rows = row_classifier.collect_all_rows()
        st.write(f"Number of rows for clustering: {len(all_rows)}")

        # Step 7: Interactive Cluster Tuning
        st.subheader("Interactive Row Cluster Tuning")
        st.write("Adjust the clustering threshold to have all lines of grade data under the same color.")
        #----- CACHED ----- (but can't hash row_classifier)
        cluster_tuning = PP.initialize_cluster_tuning(image, row_classifier, all_rows)
        #----- CACHED -----
        best_threshold = PP.compute_best_threshold(row_classifier, all_rows)

        best_threshold = st.slider("Threshold for Clustering", min_value=0.1, max_value=0.8, value=best_threshold, step=0.001)
        #----- CACHED -----
        cluster_tuning.update_clustering(best_threshold)

        # Step 8: Cluster Columns
        #----- NOT CACHED ----- (but in session state)
        grouped_data = cluster_tuning.grouped_data
        update_state('grouped_data', grouped_data)
        #----- CACHED -----
        column_clusterer = PP.load_column_clusterer(grouped_data)

        # Step 9: Process Tables into DataFrames
        st.subheader("Interactive Column Cluster Tuning")
        st.write("Adjust the clustering strength or regrouping factor to increase/decrease how words are clustered together for any table of interest.")
        #----- NOT CACHED -----
        min_samples_list = []
        for idx in range(len(grouped_data)):

            min_samples = st.slider(f"Clustering Strength for Table {idx + 1}", 
                                    min_value=1, max_value=10, 
                                    value=4, step=1, key = f"min_samples_{idx}")
            min_samples_list.append(min_samples)

        cluster_selection_epsilon = st.slider(f"Regrouping Factor (Cluster Selection Epsilon)", 
                                            min_value=0.00, max_value=200.00, 
                                            step=0.1, key = "cluster_selection_epsilon",)

        
        final_dfs = column_clusterer.process_tables_to_dataframe(grouped_data, min_samples_list, cluster_selection_epsilon, alpha = 1)
        update_state('final_dfs', final_dfs) # save to session state
        
        # Display Results
        st.subheader("Extracted Dataframes:")
        st.write("Please select grade table for analysis.")
        for idx, df in enumerate(final_dfs):
            st.write(f"\nTable {idx + 1} Preview:\n")
            st.dataframe(df, use_container_width=True)
        
        with st.expander("Download a Dataframe", expanded=False):
            df_idx = st.selectbox("Select Dataframe",np.arange(len(final_dfs))+1)
            st.download_button("Download as CSV", final_dfs[df_idx-1].to_csv(index=False), "dataframe.csv", "text/csv")

if __name__ == "__main__":
    app()
