import streamlit as st
import pandas as pd
from utils.clustering_analytics import ClusteringAnalytics
from utils.state_management import initialize_state, update_state
from utils.subject_keywords_manager import SubjectKeywordsManager
from utils.analytics_dashboard import AnalyticsDashboard

@st.cache_resource(show_spinner=False)
def import_clustering_analytics():
    return ClusteringAnalytics()

def app():
    st.title("Course Clustering Analytics")

    # Initialize session state variables
    initialize_state()

    # Check if 'final_dfs' is in session state
    if st.session_state.final_dfs is not None and len(st.session_state.final_dfs) > 0:
        df_list = st.session_state.final_dfs
        st.write("DataFrames available in session state:")
        
        # Maintain the previously selected DataFrame index
        if st.session_state.selected_df_index is not None:
            selected_df_index = st.selectbox(
                "Select a DataFrame for analysis", 
                range(len(df_list)), 
                index=st.session_state.selected_df_index, 
                format_func=lambda x: f"DataFrame {x+1}"
            )
        else:
            selected_df_index = st.selectbox(
                "Select a DataFrame for analysis", 
                range(len(df_list)), 
                format_func=lambda x: f"DataFrame {x+1}"
            )
        df = df_list[selected_df_index]
        st.write("Selected DataFrame:")
        st.dataframe(df,use_container_width=True)
        
        # Update session state with the selected DataFrame index
        update_state('selected_df_index', selected_df_index)
    else:
        df = None
        st.write("No DataFrame found in session state.")

    # Option to upload a CSV file to overwrite the current DataFrame
    with st.expander("Upload a CSV for Analysis"):
        st.write("Alternatively, you can upload a new CSV file for analysis:")
        uploaded_df = st.file_uploader("Upload a CSV file", type=["csv"])
        
        if uploaded_df is not None:
            df = pd.read_csv(uploaded_df)
            st.write("Uploaded DataFrame:")
            st.write(df)
            
            # Update session state with the uploaded file and reset selected index
            update_state('uploaded_df', uploaded_df)
            update_state('selected_df_index', None)

    # Use the previously uploaded file if available
    if st.session_state.uploaded_df is not None and uploaded_df is None:
        try:
            df = pd.read_csv(st.session_state.uploaded_df)
            st.write("Uploaded DataFrame:")
            st.dataframe(df, use_container_width=True)
        except:
            st.write("No Dataframe Uploaded.")
    
    # Initialize the ClusteringAnalytics class
    clustering_analytics = import_clustering_analytics()

    # Perform clustering if a DataFrame is loaded or uploaded
    if df is not None:
        df.columns = [str(col) for col in df.columns]
        column_name = st.selectbox("Select the column with course titles or codes", df.columns)
        type_option = st.selectbox("Select the type", ["title", "code"])

        if st.button("Classify Courses"):
            result_df = clustering_analytics.classify_courses(df, column_name, type=type_option)
            analytics_dash = AnalyticsDashboard(result_df)
            result_df = analytics_dash.df
            update_state("result_df", result_df)

    
        st.write("Classified Courses:")
        st.dataframe(st.session_state.result_df, use_container_width=True)
        if st.session_state.result_df is not None:
            st.download_button("Download Result as CSV", st.session_state.result_df.to_csv(index=False), "classified_courses.csv", "text/csv")

        with st.expander("Add New Subject or Sub-Subject", expanded=False):
            subject_manager = SubjectKeywordsManager()
            
            add_option = st.radio("What would you like to add?", ("New Subject and Courses", "Courses to Existing Subject"))
            st.write("Example Subject: Mechanical Engineering \nExample Code: APSC 100\n Example Title: Engineering Design")

            if add_option == "New Subject and Courses":
                new_subject = st.text_input("New Subject")
                new_sub_subjects = st.text_area("New Courses - separate by comma")
                subject_type = st.selectbox("Select the type for the new subject", ["title", "code"])
                

                if st.button("Add Subject and Courses"):
                    if new_subject:
                        subject_manager.add_subject(new_subject, type=subject_type)
                        if new_sub_subjects:
                            sub_subjects_list = [s.strip() for s in new_sub_subjects.split(",")]
                            for sub_subject in sub_subjects_list:
                                subject_manager.add_sub_subject(new_subject, sub_subject, type=subject_type)
                        subject_manager.save()
                        st.success(f"Successfully added '{new_subject}' with courses '{new_sub_subjects}' to {subject_type} keywords.")
                    else:
                        st.error("Subject cannot be empty.")
            
            elif add_option == "Courses to Existing Subject":
                new_sub_subjects = st.text_area("New Courses - separate by comma", key="NC1")
                subject_type = st.selectbox("Select the type for the new courses", ["title", "code"])
                existing_subjects = list(subject_manager.subject_keywords_titles.keys()) if subject_type == 'title' else list(subject_manager.subject_keywords_codes.keys())
                if existing_subjects:
                    selected_subject = st.selectbox("Select Existing Subject", existing_subjects)
                    new_sub_subjects = st.text_area("New Courses - separate by comma", key="NC2")

                    if st.button("Add Courses to Subject"):
                        if selected_subject:
                            if new_sub_subjects:
                                sub_subjects_list = [s.strip() for s in new_sub_subjects.split(",")]
                                for sub_subject in sub_subjects_list:
                                    subject_manager.add_sub_subject(selected_subject, sub_subject, type=subject_type)
                                subject_manager.save()
                                st.success(f"Successfully added courses '{new_sub_subjects}' to subject '{selected_subject}' in {subject_type} keywords.")
                            else:
                                st.error("Courses cannot be empty.")
                        else:
                            st.error("Please select an existing subject.")
                else:
                    st.error("No existing subjects found.")
            
            # Section to preview JSON files
            st.subheader("Preview JSON Files")

            preview_type = st.selectbox("Select the type of keywords to preview", ["title", "code"])

            if preview_type == "title":
                keywords_dict = subject_manager.subject_keywords_titles
            else:
                keywords_dict = subject_manager.subject_keywords_codes

            if keywords_dict:
                selected_subject = st.selectbox("Select a subject to preview", list(keywords_dict.keys()))
                st.write(f"Sub-Subjects for {selected_subject}")
                sub_subjects = keywords_dict[selected_subject]
                st.write(sub_subjects)
            else:
                st.write("No subjects found.")

        if not st.session_state.result_df.empty:
            analytics_dash = AnalyticsDashboard(st.session_state.result_df)
            cleaned_df = analytics_dash.df
            analytics_dash.display()
