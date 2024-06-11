import streamlit as st
from PIL import Image
from utils.state_management import initialize_state, update_state
import utils.pipeline_processor as PP
# import numpy as np

@st.cache_data(show_spinner=False)
def predict_image(image_path, iou, conf, agnostic_nms):
    return vision_pipeline.predict(image_path, iou=iou, conf=conf, agnostic_nms=agnostic_nms)

@st.cache_data(show_spinner=False)
def display_image_with_boxes(image_path, boxes, classes):
    img_with_boxes_path = vision_pipeline.visualize_boxes(image_path, boxes, classes, vision_pipeline.class_names)
    img_with_boxes = Image.open(img_with_boxes_path)
    return img_with_boxes

# @st.experimental_fragment
def app(): 
    # update_state("page", "Extraction")
    global vision_pipeline
    st.title("Information Extraction - Table Reconstruction Pipeline")
    
    # Initialize session state
    initialize_state()
    
    # Step 1.5: Upload Image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], key='file_uploader')
    if uploaded_file is not None:
        update_state('uploaded_file', uploaded_file)

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
        col1, col2 = st.columns(2)
        col1.metric("Number of cropped images", len(cropped_images))

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
        col2.metric("Number of rows for clustering", len(all_rows))

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
        if headers:
            with st.expander("Preview Extracted List of Possible Table Headers"):
                st.subheader("Preview Table Headers:")
                header_previews = PP.headers_to_strings(headers)
                for idx, header_text in enumerate(header_previews):
                    st.markdown(f"**Header {idx + 1} Preview:**\n```\n{header_text}\n```")
        else:
            st.write("No Headers Found.")

        with st.expander("Preview Extracted List of Possible Table Data"):
            st.subheader("Preview Row-Clustered Tables:")
            table_previews = PP.groups_to_strings(grouped_data)
            for idx, table_text in enumerate(table_previews):
                st.markdown(f"**Table {idx + 1} Preview:**\n```\n{table_text}\n```")

        # Step 9: Select Header & Table for LLM Inference
        if headers:
            st.subheader("Select Header for LLM Inference")
            selected_header_idx = st.selectbox("Select Header", [i+1 for i, _ in enumerate(header_previews)])
            selected_header_text = header_previews[selected_header_idx-1]
        else:
            st.write("No Headers Found.")

        st.subheader("Select Table for LLM Inference")
        selected_table_idx = st.selectbox("Select Table", [i+1 for i, _ in enumerate(table_previews)])
        selected_table_text = table_previews[selected_table_idx-1]
        selected_table_nlines = selected_table_text.count('\n') + 1 # Count number of lines for consistency check
        update_state("string_nlines", selected_table_nlines)

        # Select Mistral Model Path
        mistral_model_path = st.text_input("LLM Model Path", value=st.session_state.llm_path)
        update_state("llm_path",mistral_model_path)

        # Step 10: Query Mistral
        if st.button("Query Mistral"):
            stripped_selected_table = selected_table_text.replace(',','')
            stripped_selected_headers = selected_header_text.replace(',','')
            final_df = PP.query_mistral(stripped_selected_headers if headers else None, stripped_selected_table, mistral_model_path)
            update_state("final_df", final_df)

            final_df_nlines = final_df.shape[0]
            update_state("df_nlines", final_df_nlines)
            
        # Display the dataframe if it exists
        if 'final_df' in st.session_state and st.session_state.final_df is not None:
            st.subheader("Extracted Dataframe using LLM:")

            # Perform Consistency Check between # of courses in the string and # of courses in the DF
            if st.session_state.df_nlines is not None and st.session_state.string_nlines is not None:
                string_row_count, df_row_count = st.columns(2, gap="small")
                string_row_count.metric("Row Count from OCR", st.session_state.string_nlines)
                df_row_count.metric("Row Count from Extracted Data", st.session_state.df_nlines)
                is_consistent = selected_table_nlines == final_df_nlines
                if is_consistent:
                    st.success("All rows are accounted for!")
                else:
                    st.error("Row count mismatch! Please check the data.")

            st.write("Below is the result from querying Mistral. You can download and edit the table as needed.")
            edited_df = st.data_editor(st.session_state.final_df)
            update_state("final_df", edited_df)

            # Save the updated dataframe
            if st.button("Save DataFrame"):
                st.session_state.final_df.to_csv("grade_dataframe.csv", index=False)
                st.success("DataFrame saved successfully!")
        else:
            st.write("Query Mistral to process dataframe.")
        

if __name__ == "__main__":
    app()
