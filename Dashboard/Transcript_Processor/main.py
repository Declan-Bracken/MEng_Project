import streamlit as st
import pages.extraction as extraction
# import pages.analytics as analytics
from utils.state_management import initialize_state, update_state

# Set the page configuration to use wide mode
st.set_page_config(layout="wide")

initialize_state()
# Sidebar navigation
def navigate():
    st.session_state.page = st.session_state.navigation

# with st.sidebar:
#     st.radio("Go to", ["Main", "Extraction"], key='navigation', on_change=navigate) #, "Analytics"

# Define the main page
def main_page():
    st.title("Transcript Vision")
    st.write("Developed by Declan Bracken")
    st.header("Welcome to Transcript Vision")
    st.write("""
    Transcript Vision is an innovative tool designed to simplify and enhance the process of extracting and analyzing university student transcript information. Our app provides a seamless experience for administrators, educators, and students to gain valuable insights from academic records with ease and accuracy.
    
    ### How It Works
    For now, this Transcript Vision prototype only works for data extraction and not post-extraction analytics:
    
    **Extraction Module**: This module allows users to upload and extract information from student transcripts. Our advanced algorithms accurately identify and capture relevant data, making the extraction process efficient and pain-free.
    
    ### Get Started
    Use the navigation panel to select the extraction module.
    """)
# 2. **Analytics Module**: Once the data is extracted, the Analytics module offers powerful tools to analyze the information. Users can classify courses into predetermined categories, generate detailed reports, visualize academic trends, and gain insights into student performance, helping in decision-making and academic planning.

# Page mapping
pages = {
    "Main": main_page,
    "Extraction": extraction.app,
    # "Analytics": analytics.app
}

# Load the selected page
if st.session_state.page in pages:
    pages[st.session_state.page]()
else:
    main_page()
