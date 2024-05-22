import streamlit as st
import pandas as pd
from scipy.stats import zscore
import altair as alt
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import xlsxwriter

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
        self.fix_pluses()
        self.map_grades()

    def clean_data(self):
        # Strip whitespace and remove unwanted characters from all cells and column headers
        self.df = self.df.replace({r'[;:"<>~`@!*]': ''}, regex=True)
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    def fix_pluses(self, grade_column = 'Grade'):
        plus_mapping = {'At': 'A+', 'Bt': 'B+', 'Ct': 'C+', 'Dt': 'D+'}
        self.df[grade_column] = self.df['Grade'].map(plus_mapping).fillna(self.df[grade_column])

    def map_grades(self, grade_column = 'Grade'):
        # Ensure the grade column is in the dataframe
        if grade_column not in self.df.columns:
            raise ValueError(f"The specified grade column '{grade_column}' is not found in the dataframe.")

        # Map the grades
        self.df['Numeric Grade'] = self.df[grade_column].map(self.GRADE_MAPPING).fillna(self.df[grade_column])
        # Convert 'Numeric Grade' to numeric values, coercing errors to NaN
        self.df['Numeric Grade'] = pd.to_numeric(self.df['Numeric Grade'], errors='coerce')

    def compute_metrics(self):
        # Compute necessary metrics     
        avg_grade = self.df['Numeric Grade'].mean()
        num_courses = self.df['Course Code'].nunique()
        total_credits = self.df['Credits'].sum()

        # Calculate GPA
        weighted_grades = self.df['Numeric Grade'] * self.df['Credits']
        gpa = weighted_grades.sum() / total_credits if total_credits != 0 else float('nan')
        
        return avg_grade, num_courses, total_credits, gpa
    
    def plot_grade_distribution(self):
        # Prepare the data
        df = self.df[['Course Code', 'Numeric Grade', 'cluster']].dropna()
        
        # Create the Altair line chart
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('Course Code:N', axis=alt.Axis(labelAngle=-45, title='Course Code')),
            y=alt.Y('Numeric Grade:Q', axis=alt.Axis(title='Numeric Grade')),
            tooltip=['Course Code', 'Numeric Grade']
        ).properties(
            width='container',
            height=400,
            # title='Grade Distribution'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16
        )
        
        # Display the chart in Streamlit
        # st.altair_chart(chart, use_container_width=True)
        return chart
    
    # def plot_credits_distribution(self):
    #     # Group by cluster and sum the credits
    #     grouped = self.df.groupby('cluster')[['Credits']].sum().reset_index().sort_values(by='Credits')

    #     # Create the Altair bar chart
    #     chart = alt.Chart(grouped).mark_bar().encode(
    #         x=alt.X('cluster:N', axis=alt.Axis(title='Course Type', labelAngle=-45)),
    #         y=alt.Y('Credits:Q', axis=alt.Axis(title='Total Credits')),
    #         tooltip=['cluster', 'Credits']
    #     ).properties(
    #         width='container',
    #         height=400,
    #         # title='Credits Distribution by Course Type'
    #     ).configure_axis(
    #         labelFontSize=12,
    #         titleFontSize=14
    #     ).configure_title(
    #         fontSize=16
    #     )
        
    #     # Display the chart in Streamlit
    #     st.altair_chart(chart, use_container_width=True)
        
    def plot_credits_distribution(self):
        # Group by cluster and sum the credits
        grouped = self.df.groupby('cluster')[['Credits']].sum().reset_index().sort_values(by='Credits')

        # Create the Altair pie chart
        chart = alt.Chart(grouped).mark_arc().encode(
            theta=alt.Theta(field='Credits', type='quantitative'),
            color=alt.Color(field='cluster', type='nominal', legend=alt.Legend(title="Course Type")),
            tooltip=['cluster', 'Credits']
        ).properties(
            width=400,
            height=400,
            # title='Credit Distribution by Subject'
        ).configure_legend(
            titleFontSize=17,
            labelFontSize=15
        ).configure_title(
            fontSize=25
        )
        
        # Display the chart in Streamlit
        # st.altair_chart(chart, use_container_width=True)
        return chart
    
    # def plot_credits_distribution(self):
    #     # Group by cluster and sum the credits
    #     grouped = self.df.groupby('cluster')[['Credits']].sum().reset_index().sort_values(by='Credits')

    #     # Create the Altair pie chart
    #     chart = alt.Chart(grouped).mark_arc(outerRadius=120).encode(
    #         theta=alt.Theta(field='Credits', type='quantitative'),
    #         color=alt.Color(field='cluster', type='nominal', legend=alt.Legend(title="Course Type")),
    #         tooltip=['cluster', 'Credits'])


    #     # Create text annotations
    #     text = chart.mark_text(radius=150, size=14).encode(text="cluster:N")

    #     final_chart = chart + text

    #     # Display the chart in Streamlit
    #     st.altair_chart(final_chart, use_container_width=True)


    def detect_anomalies(self):
        # Detect grade anomalies using Z-score
        numeric_grades = self.df['Numeric Grade']

        # Create a mask for NaN values
        nan_mask = numeric_grades.isna()

        # Calculate Z-scores for non-NaN values
        self.df['Z-Score'] = zscore(numeric_grades.dropna())
        
        # Mark NaN values as anomalies
        self.df.loc[nan_mask, 'Z-Score'] = float('inf')
        
        # Filter anomalies
        anomalies = self.df[(self.df['Z-Score'].abs() > 2) | nan_mask]
        return anomalies
    
    def detect_failed_courses(self):
        # Detect failed grades (assuming grades below a certain threshold are failures)
        # Here, we consider grades below 1.0 as failed grades
        failed_grades = self.df[self.df['Numeric Grade'] < 1.0]
        return failed_grades

    def find_best_courses(self):
        try:
            unique_grades = self.df['Numeric Grade'].dropna().unique()
            unique_grades.sort()
            best_grade = unique_grades[-1] #take highest grade
            # Filter the DataFrame to include only the rows with these top 3 grades
            best_courses = self.df[self.df['Numeric Grade'] == best_grade]
            return best_courses
        except Exception as e:
            print(e)
            return None

    def plot_course_type_distribution(self):
        # Group by course type and calculate the mean grade
        grouped = self.df.groupby('cluster')[['Numeric Grade']].mean().reset_index().sort_values(by='Numeric Grade')

        # Create the Altair bar chart
        chart = alt.Chart(grouped).mark_bar().encode(
            x=alt.X('cluster:N', axis=alt.Axis(title='Course Type', labelAngle=-45)),
            y=alt.Y('Numeric Grade:Q', axis=alt.Axis(title='Average Numeric Grade')),
            tooltip=['cluster', 'Numeric Grade']
        ).properties(
            width='container',
            height=400,
            # title='Average Grade by Course Type'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16
        )
        
        # Display the chart in Streamlit
        # st.altair_chart(chart, use_container_width=True)
        return chart

    def display(self):
        # Display analytics and visualizations
        st.title("Candidate Strength Assessment")

        # Compute and display metrics
        avg_grade, num_courses, total_credits, gpa = self.compute_metrics()
        st.subheader("Metrics")
        # st.write(f"Average Grade: {avg_grade:.2f}")
        # st.write(f"Number of Courses: {num_courses}")
        # st.write(f"Total Credits: {total_credits}")
        # st.write(f"Calculated CGPA: {gpa:.2f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Calculated CGPA", f"{gpa:.2f}")
        col2.metric("Average Grade", f"{avg_grade:.2f}")
        col3.metric("Number of Courses", num_courses)
        col4.metric("Total Credits", total_credits)
        
        # Plot and display visualizations
        # Plot and display visualizations
        st.subheader("Grade Distribution")
        grade_chart = self.plot_grade_distribution()
        st.altair_chart(grade_chart, use_container_width=True)

        st.subheader("Average Grade by Subject")
        course_type_chart = self.plot_course_type_distribution()
        st.altair_chart(course_type_chart, use_container_width=True)

        st.subheader("Credits Distribution by Subject")
        credits_chart = self.plot_credits_distribution()
        st.altair_chart(credits_chart, use_container_width=True)

        # Best Marks:
        st.subheader("Best Performances")
        best_marks = self.find_best_courses()
        st.write(best_marks if not best_marks is None else "Could not find the highest unique grade")

        # Display anomalies and failed grades
        st.subheader("Worst Performances & Anomalies")
        anomalies = self.detect_anomalies()
        st.write(anomalies if not anomalies.empty else "No anomalies detected")

        st.subheader("Failed Courses")
        failed_grades = self.detect_failed_courses()
        st.write(failed_grades if not failed_grades.empty else "No failed grades detected")

        # Add a button to save the data and charts to an Excel file
        if st.button("Save to Excel"):
            self.save_to_excel(avg_grade, num_courses, total_credits, gpa, grade_chart, course_type_chart, credits_chart, anomalies, failed_grades)
        
    def save_to_excel(self, avg_grade, num_courses, total_credits, gpa, grade_chart, course_type_chart, credits_chart, anomalies, failed_grades):
        with pd.ExcelWriter('candidate_assessment.xlsx', engine='xlsxwriter') as writer:
            metrics_df = pd.DataFrame({
                'Metric': ['Average Grade', 'Number of Courses', 'Total Credits', 'Calculated CGPA'],
                'Value': [f"{avg_grade:.2f}", num_courses, total_credits, f"{gpa:.2f}"]
            })
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            self.df.to_excel(writer, sheet_name='Grades', index=False)
            anomalies.to_excel(writer, sheet_name='Anomalies', index=False)
            failed_grades.to_excel(writer, sheet_name='Failed Grades', index=False)

            workbook = writer.book
            worksheet = workbook.add_worksheet('Charts')

            # Save charts as HTML and embed in Excel
            def save_chart_as_html(chart, filename):
                with open(filename, 'w') as f:
                    f.write(chart.to_html())

            save_chart_as_html(grade_chart, 'grade_distribution.html')
            save_chart_as_html(course_type_chart, 'course_type_distribution.html')
            save_chart_as_html(credits_chart, 'credits_distribution.html')

            worksheet.write('A1', 'Grade Distribution')
            worksheet.insert_image('A2', 'grade_distribution.html', {'x_scale': 0.5, 'y_scale': 0.5})
            worksheet.write('A22', 'Course Type Distribution')
            worksheet.insert_image('A23', 'course_type_distribution.html', {'x_scale': 0.5, 'y_scale': 0.5})
            worksheet.write('A43', 'Credits Distribution')
            worksheet.insert_image('A44', 'credits_distribution.html', {'x_scale': 0.5, 'y_scale': 0.5})

        st.success("Data and charts have been saved to candidate_assessment.xlsx")

