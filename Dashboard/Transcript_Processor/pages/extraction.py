import streamlit as st
from PIL import Image
from utils.state_management import initialize_state, update_state
import utils.pipeline_processor as PP

@st.cache_data(show_spinner=False)
def predict_image(image_path, iou, conf, agnostic_nms):
    return vision_pipeline.predict(image_path, iou=iou, conf=conf, agnostic_nms=agnostic_nms)

@st.cache_data(show_spinner=False)
def display_image_with_boxes(image_path, boxes, classes):
    img_with_boxes_path = vision_pipeline.visualize_boxes(image_path, boxes, classes, vision_pipeline.class_names)
    img_with_boxes = Image.open(img_with_boxes_path)
    return img_with_boxes
    # return boxes, classes

@st.experimental_fragment
def app():
    global vision_pipeline
    st.title("Information Extraction - Table Reconstruction Pipeline")
    
    # Initialize session state
    initialize_state()

    # Step 1: Set Vision Model Path
    st.subheader("Set Vision Model Path")
    model_path = st.text_input("Vision Model Path", value=st.session_state.model_path)
    if st.button("Set Model Path"):
        update_state('model_path', model_path) # Update status of vision model path
        update_state('vision_model_loaded', False) # Update status of vision model
        st.success("Changed Model Path!")
    
    # Step 1.5: Upload Image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], key='file_uploader')
    if uploaded_file is not None:
        update_state('uploaded_file', uploaded_file)

    if st.session_state.uploaded_file is not None:

        # Clear cache when a new file is uploaded
        if st.session_state.uploaded_file != uploaded_file:
            st.cache_data.clear()

        #----- CACHED -----
        image_path = PP.upload_image(st.session_state.uploaded_file)

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
        #----- NOT CACHED -----
        min_samples_list = []
        for idx in range(len(grouped_data)):
            min_samples = st.slider(f"Clustering Strength for Table {idx + 1}", min_value=1, max_value=10, value=4, step=1)
            min_samples_list.append(min_samples)

        #----- NOT CACHED -----
        final_dfs = column_clusterer.process_tables_to_dataframe(grouped_data, min_samples_list)
        update_state('final_dataframes', final_dfs)

        # Display Results
        st.subheader("Extracted Dataframes:")
        st.write("Please select grade table for analysis.")
        for idx, df in enumerate(final_dfs):
            st.write(f"\nTable {idx + 1} Preview:\n")
            #----- CACHED -----
            PP.display_aggrid(df, key=f"table_{idx}")

if __name__ == "__main__":
    app()
