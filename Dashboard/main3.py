import streamlit as st
from PIL import Image
import tempfile
import sys
import torch
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pandas as pd

# Add your parent directory to the path
parent_dir = r'c:\Users\Declan Bracken\MEng_Project'
sys.path.append(parent_dir)
from Pipelines.Py_files.mistral_pipeline_v2 import MistralInference
from Pipelines.Py_files.vision_pipeline_stlit import VisionPipeline
from TableReconstruction.image_processor import ImageProcessor
from TableReconstruction.text_classifier import TextClassifier
from TableReconstruction.row_clusterer_v2 import RowClassifier
from TableReconstruction.streamlit_cluster_tuning import StreamlitClusterTuning
from Dashboard.analytics_dashboard import AnalyticsDashboard

@st.cache_data(show_spinner=False)
def set_default_model_path(new_path):
    st.session_state.default_model_path = new_path

@st.cache_data(show_spinner=False)
def upload_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    return temp_file_path

@st.cache_data(show_spinner=False)
def predict_image(image_path, iou, conf, agnostic_nms):
    results = pipeline.predict(image_path, iou=iou, conf=conf, agnostic_nms=agnostic_nms)
    return results

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

@st.cache_resource(show_spinner=False)
def classify_text(cropped_images, classes, boxes):
    text_classifier = TextClassifier(cropped_images, classes, boxes)
    return text_classifier

@st.cache_resource(show_spinner=False)
def initialize_cluster_tuning(image, _row_classifier, all_rows):
    return StreamlitClusterTuning(image, _row_classifier, all_rows)

@st.cache_data(show_spinner=False)
def compute_best_threshold(_row_classifier, all_rows):
    best_threshold = _row_classifier.optimize_by_histogram(all_rows)
    return best_threshold

@st.cache_resource(show_spinner=False)
def cluster_rows(_text_classifier):
    row_classifier = RowClassifier(_text_classifier.headers, _text_classifier.single_row, _text_classifier.tables)
    all_rows = row_classifier.collect_all_rows()
    return row_classifier, all_rows

@st.cache_resource
def load_vision_pipeline(model_path):
    return VisionPipeline(model_path)

@st.cache_resource
def load_mistral(model_path=r"C:\Users\Declan Bracken\MEng_Project\mistral\models\dolphin-2.1-mistral-7b.Q5_K_M.gguf"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mistral_pipeline = MistralInference(device=device, model_path=model_path)
    return mistral_pipeline

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
def display_aggrid(df, key):
    df.columns = [str(col) for col in df.columns]  # Ensure all column names are strings
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)  # Enable pagination
    gb.configure_default_column(editable=True, resizable=True, groupable=True)  # Enable editable and resizable columns
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)  # Enable column selection

    gridOptions = gb.build()
    row_height = 29  # Default row height
    header_height = 56  # Default header height
    total_rows = len(df)
    grid_height = max(header_height + (total_rows * row_height), 100)  # Ensure a minimum height

    grid_response = AgGrid(
        df, 
        gridOptions=gridOptions, 
        height=grid_height, 
        fit_columns_on_grid_load=True, 
        update_mode=GridUpdateMode.MODEL_CHANGED, 
        key=key
    )

    # selected_columns = [col for col in grid_response['selected_columns']] if 'selected_columns' in grid_response else []
    # updated_df = grid_response['data']
    return grid_response #selected_columns, pd.DataFrame(updated_df)

@st.cache_data
def headers_to_strings(header_lines):
    strings = []
    for headers in header_lines:
        ordered_lines = headers["text"]["text"]
        strings.append(' '.join(ordered_lines).strip())
    return strings

# Initialize the global variable for the vision pipeline
pipeline = None
# Load or set default model path
if 'default_vision_model_path' not in st.session_state:
    st.session_state.default_model_path = r'C:\Users\Declan Bracken\MEng_Project\yolo_training\yolo_v8_models\finetune_v4 (3_classes)\best (1).pt'

def main_mistral():
    global pipeline
    # Create a sidebar navigation
    st.sidebar.title("Navigation")
    pages = ["Information Extraction", "Candidate Strength Assessment"]
    choice = st.sidebar.radio("Go to", pages)

    if choice == "Information Extraction":
        st.title("Information Extraction - Table Reconstruction Pipeline")
        
        # Step 1: Set Vision Model Path
        model_path = st.text_input("Vision Model Path", value=st.session_state.default_model_path)
        if st.button("Set as Default Path"):
            set_default_model_path(model_path)
            st.success("Default model path updated!")

        # Load Vision Pipeline
        pipeline = load_vision_pipeline(model_path)
        
        # Step 1.5: Upload Image
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file:
            if "uploaded_file" in st.session_state and st.session_state.uploaded_file != uploaded_file:
                st.cache_data.clear()
                st.session_state.dataframes = None
                st.session_state.selected_columns = {}
                st.session_state.final_df = None

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
            headers = text_classifier.headers
            
            # Step 6: Cluster the Rows
            row_classifier, all_rows = cluster_rows(text_classifier)
            st.write(f"Number of rows for clustering: {len(all_rows)}")

            # Step 7: Interactive Cluster Tuning
            st.subheader("Interactive Row Cluster Tuning")
            cluster_tuning = initialize_cluster_tuning(image, row_classifier, all_rows)
            best_threshold = compute_best_threshold(row_classifier, all_rows)
            best_threshold = st.slider("Threshold for Clustering", min_value=0.1, max_value=0.8, value=best_threshold, step=0.001)
            cluster_tuning.update_clustering(best_threshold)

            # Step 8: Preview Row-Clustered Tables and Headers
            grouped_data = cluster_tuning.grouped_data
            if headers:
                st.subheader("Preview Table Headers:")
                header_previews = headers_to_strings(headers)
                for idx, header_text in enumerate(header_previews):
                    st.markdown(f"**Header {idx + 1} Preview:**\n```\n{header_text}\n```")
            else:
                st.write("No Headers Found.")

            st.subheader("Preview Row-Clustered Tables:")
            table_previews = groups_to_strings(grouped_data)
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

            # Select Mistral Model Path
            mistral_model_path = st.text_input("LLM Model Path", value=r"C:\Users\Declan Bracken\MEng_Project\mistral\models\dolphin-2.1-mistral-7b.Q5_K_M.gguf")

            # Step 10: Query Mistral
            if st.button("Query Mistral"):
                st.write("Querying Mistral...")
                mistral_pipeline = load_mistral(mistral_model_path)
                if headers:
                    final_df = mistral_pipeline.process_transcript(selected_header_text, selected_table_text)
                else:
                    final_df = mistral_pipeline.process_transcript(None, selected_table_text)
                st.write("LLM Inference Result:")
                # Store the resulting dataframe in session state
                st.session_state.final_df = final_df

                st.subheader("LLM Inference Result")
                st.write("Below is the result from querying Mistral. You can edit the table as needed.")

                # Display the resulting dataframe in an editable AgGrid
                updated_df = display_aggrid(final_df, key='mistral_table')

                # Save the updated dataframe
                if st.button("Save Updated DataFrame"):
                    updated_df.to_csv("updated_transcript.csv", index=False)
                    st.success("Updated DataFrame saved successfully!")

    elif choice == "Candidate Strength Assessment":
        st.title("Candidate Strength Assessment")

        # Step 1: Upload CSV or use existing dataframe
        uploaded_csv = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
        elif 'final_df' in st.session_state:
            df = st.session_state.final_df
        else:
            st.warning("Please upload a CSV file or complete the Information Extraction phase to load a dataframe.")
            return

        # Initialize the Analytics Dashboard with the dataframe
        analytics_dashboard = AnalyticsDashboard(df)
        analytics_dashboard.display()

if __name__ == "__main__":
    main_mistral()