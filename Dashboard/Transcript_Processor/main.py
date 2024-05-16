import streamlit as st
import pages.extraction as extraction
import pages.analytics as analytics
# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Extraction'

# Load the selected page
if st.session_state.page == "Extraction":
    extraction.app()
elif st.session_state.page == "Analytics":
    analytics.app()
