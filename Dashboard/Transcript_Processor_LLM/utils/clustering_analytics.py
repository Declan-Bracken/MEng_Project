from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
sys.path.append(os.path.dirname(__file__))

from subject_keywords_manager import SubjectKeywordsManager

class ClusteringAnalytics:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', titles_path='Dashboard/Transcript_Processor/subjects/subject_keywords_titles.json', codes_path='Dashboard/Transcript_Processor/subjects/subject_keywords_codes.json'):
        self.model = SentenceTransformer(embedding_model)
        self.manager = SubjectKeywordsManager(titles_path, codes_path)

    def classify_courses(self, df, column_name, type='title'):
        # Load the subject keywords
        if type == 'title':
            subject_keywords = self.manager.subject_keywords_titles
        elif type == 'code':
            subject_keywords = self.manager.subject_keywords_codes
        else:
            raise ValueError("Type must be 'title' or 'code'")

        # Compute embeddings for the keywords
        subject_embeddings = {subject: self.model.encode(keywords) for subject, keywords in subject_keywords.items()}

        # Compute embeddings for the course titles or codes
        course_texts = df[column_name].tolist()
        course_embeddings = self.model.encode(course_texts)

        # Function to compute the closest subject for each course
        def assign_courses_to_subjects(course_embeddings, subject_embeddings):
            course_subjects = []
            for course_embedding in course_embeddings:
                best_subject = None
                best_similarity = -1
                for subject, embeddings in subject_embeddings.items():
                    # Compute cosine similarity between the course embedding and all keyword embeddings for the subject
                    similarities = cosine_similarity([course_embedding], embeddings).flatten()
                    max_similarity = np.max(similarities)
                    if max_similarity > best_similarity:
                        best_similarity = max_similarity
                        best_subject = subject
                course_subjects.append(best_subject)
            return course_subjects

        # Assign courses to subjects
        assigned_subjects = assign_courses_to_subjects(course_embeddings, subject_embeddings)

        # Add the assigned subjects as a new column in the DataFrame
        df['cluster'] = assigned_subjects
        return df

# Example usage:
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        'course_title': [
            'Introduction to Writing and Communication', 'Introduction to Philosophy', 'Counseling Psychology I', 
            'Counseling Psychology II', 'Forest Ecology and Management', 'Advanced German Literature', 
            'Advanced Calculus', 'Introduction to Earth Science', 'Linear Algebra', 'Mathematical Analysis', 
            'Complex Variables', 'Ethics in Philosophy', 'Conservation and Sustainability', 'East Asian History', 
            'Introduction to Statistics', 'Advanced Counseling Skills', 'Clinical Supervision', 'Ordinary Differential Equations', 
            'Discrete Mathematics', 'Applied Statistics', 'Introduction to Computer Science', 'Number Theory', 
            'Real Analysis', 'Introductory Physics', 'Introduction to Business', 'Introduction to Computer Science', 
            'English for Multilingual Speakers', 'Calculus I', 'Calculus II', 'Data Structures', 
            'Introduction to Economics', 'Algebra and Geometry', 'Calculus III', 'Introduction to Probability', 
            'Microeconomics', 'Macroeconomics', 'Calculus I for Engineers', 'Calculus II for Engineers', 
            'General Physics I', 'General Physics II', 'Introduction to Computer Programming', 'Academic Reading Skills', 
            'Basic Computer Skills', 'Linear Algebra for Engineers', 'Introduction to Business Concepts', 
            'Fundamentals of Physical Fitness', 'Organizational Behaviour', 'Financial Accounting', 'Business Analytics', 
            'International Business', 'Operations Management'
        ]
    }
    df = pd.DataFrame(data)
    
    # Initialize the ClusteringAnalytics class
    clustering_analytics = ClusteringAnalytics()

    # Classify the courses in the DataFrame
    result_df = clustering_analytics.classify_courses(df, 'course_title', type='title')
    
    # Print the result
    print(result_df)
