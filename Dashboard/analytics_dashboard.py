import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

class AnalyticsDashboard:
    def __init__(self, df):
        self.df = df
        self.GRADE_MAPPING = {
            'A+': 4.3, 'A': 4.0, 'A-': 3.7,
            'B+': 3.3, 'B': 3.0, 'B-': 2.7,
            'C+': 2.3, 'C': 2.0, 'C-': 1.7,
            'D+': 1.3, 'D': 1.0, 'D-': 0.7,
            'F': 0.0
        }
        self.clean_data()
        self.map_grades()
    
    def clean_data(self):
        # Strip whitespace and remove unwanted characters from all cells and column headers
        self.df = self.df.replace({r'[;:"]': ''}, regex=True)
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    def map_grades(self, grade_column = 'Grade'):
        # Ensure the grade column is in the dataframe
        if grade_column not in self.df.columns:
            raise ValueError(f"The specified grade column '{grade_column}' is not found in the dataframe.")

        # Map the grades
        self.df['Numeric Grade'] = self.df[grade_column].map(self.GRADE_MAPPING).fillna(self.df[grade_column])

    
    def compute_metrics(self):
        # Compute necessary metrics     
        # self.df['Numeric Grade'] = self.df[self.df['Grade'].map(self.grade_mapping).notna()] #pd.to_numeric(self.df['Grade'], errors='coerce')
        avg_grade = self.df['Numeric Grade'].mean()
        num_courses = self.df['Course Code'].nunique()
        total_credits = self.df['Credits'].sum()
        
        return avg_grade, num_courses, total_credits
    
    def plot_grade_distribution(self):
        # Plot grade distribution
        fig, ax = plt.subplots()
        sns.histplot(self.df['Numeric Grade'].dropna(), kde=True, ax=ax)
        ax.set_title('Grade Distribution')
        ax.set_xlabel('Grade')
        ax.set_ylabel('Frequency')
        return fig
    
    def plot_credits_distribution(self):
        # Plot credits distribution per course
        fig, ax = plt.subplots()
        sns.histplot(self.df['Credits'], kde=True, ax=ax)
        ax.set_title('Credits Distribution')
        ax.set_xlabel('Credits')
        ax.set_ylabel('Frequency')
        return fig

    def detect_anomalies(self):
        # Detect grade anomalies using Z-score
        self.df['Z-Score'] = zscore(self.df['Numeric Grade'].dropna())
        anomalies = self.df[(self.df['Z-Score'].abs() > 2)]
        return anomalies
    
    def detect_failed_grades(self):
        # Detect failed grades (assuming grades below a certain threshold are failures)
        # Here, we consider grades below 50 as failed grades
        failed_grades = self.df[self.df['Numeric Grade'] < 50]
        return failed_grades
    
    def plot_course_type_distribution(self):
        # Group by course type and plot grade distribution
        self.df['Course Type'] = self.df['Course Code'].str.extract(r'([A-Za-z]+)', expand=False)
        grouped = self.df.groupby('Course Type')['Numeric Grade'].mean().sort_values()
        
        fig, ax = plt.subplots()
        grouped.plot(kind='bar', ax=ax)
        ax.set_title('Average Grade by Course Type')
        ax.set_xlabel('Course Type')
        ax.set_ylabel('Average Grade')
        return fig

    def display(self):
        # Display analytics and visualizations
        st.title("Candidate Strength Assessment")

        # Compute and display metrics
        avg_grade, num_courses, total_credits = self.compute_metrics()
        st.subheader("Metrics")
        st.write(f"Average Grade: {avg_grade:.2f}")
        st.write(f"Number of Courses: {num_courses}")
        st.write(f"Total Credits: {total_credits}")

        # Plot and display visualizations
        st.subheader("Visualizations")
        st.write("### Grade Distribution")
        st.pyplot(self.plot_grade_distribution())

        st.write("### Credits Distribution")
        st.pyplot(self.plot_credits_distribution())

        # Display anomalies and failed grades
        st.subheader("Grade Anomalies")
        anomalies = self.detect_anomalies()
        st.write(anomalies if not anomalies.empty else "No anomalies detected")

        st.subheader("Failed Grades")
        failed_grades = self.detect_failed_grades()
        st.write(failed_grades if not failed_grades.empty else "No failed grades detected")

        # Plot grade distribution by course type
        st.subheader("Average Grade by Course Type")
        st.pyplot(self.plot_course_type_distribution())
