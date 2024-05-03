from llama_cpp import Llama
import pandas as pd
from io import StringIO
import torch
import gc
import os
import sys
import csv

class SuppressOutput:
    def __enter__(self):
        # Redirect stdout and stderr to os.devnull
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the devnull handlers and restore original handlers
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class MistralInference():
    _instance = None  # Class variable to store the singleton instance

    # Create new class instance if one doesn't exist (Singleton pattern)
    def __new__(cls, device=torch.device('cuda')):
        if cls._instance is None:
            print("Creating new instance of MistralInference")
            cls._instance = super(MistralInference, cls).__new__(cls)

            # Initialize instance variables
            cls._instance.initialize(device)
        else:
            print("Using existing instance of MistralInference")

        return cls._instance
    
    def initialize(self, device, model_path = r"C:\Users\Declan Bracken\MEng_Project\mistral\models\dolphin-2.1-mistral-7b.Q5_K_M.gguf", max_sequence_len = 4096):
        self.headers = ['Course Code', 'Grade', 'Credits']
        self.model_name = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"
        if device == torch.device('cuda') and not torch.cuda.is_available():
            print("Switching device to CPU.")
            device = torch.device('cpu')

        torch.cuda.empty_cache()
        gc.collect()

        # Dynamically determine optimal number of threads and GPU layers
        n_threads = max(1, torch.get_num_threads() - 1)  # Leave one thread for OS and other processes
        print(f"Using {n_threads} CPU threads.")
        n_gpu_layers = 0  # Default to no GPU layers if not using CUDA
        if device.type == 'cuda':
            n_gpu_layers = torch.cuda.get_device_capability(device.index)[0]  # Use number of SMs as a heuristic
            print(f"Offloading {n_gpu_layers} layers to GPU.")

        print(f"""-----------------------------------------------------------------------\nLoading Mistral to {device}...\n-----------------------------------------------------------------------""")
        with SuppressOutput(): # Suppress Verbose Output
            try:
                self.llm = Llama(
                model_path= model_path,     # Download the model file first
                n_ctx=max_sequence_len,     # The max sequence length to use - note that longer sequence lengths require much more resources
                n_threads=n_threads,        # The number of CPU threads to use, tailor to your system and the resulting performance
                n_gpu_layers=n_gpu_layers   # The number of layers to offload to GPU, if you have GPU acceleration available
                )
                print("Model Loaded Successfully!")
            except Exception as e:
                print(f"Error Loading Model Weights: {e}")
            
    def query_mistral(self, headers, text):
        prompt = f'''
Below is OCR text from a student transcript. This text contains a table, or multiple tables. Select data only relevant to student courses and grades from these tables and format the fields into a table in csv format. Some extracted table headers are given below to help with formatting. The csv you output should only have 3 columns: 'Course Code', 'Grade', and 'Credits', you must select which columns best fit these fields.
        
### Headers:
{headers}

### Text:
{text}

### CSV:

'''

        system_message = "You are a table creation assistant"
        prompt_template=f'''<|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant
        '''

        max_tokens = 2048
        temperature = 0
        echo = False
        stop = ["</s>"]
        try:
            print("Prompting Mistral...")
            # Define the parameters
            with SuppressOutput():
                model_output = self.llm(
                    prompt_template,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    echo=echo,
                    stop=stop,
                )
            print("Prompt Successful!")
            final_result = model_output["choices"][0]["text"].strip()

            return final_result
        
        except Exception as e:
            print(f"An error occurred during model inference: {e}")

    def filter_unidentified_courses(self, mistral_output, original_text):
        # Step 1: Extract course codes from Mistral's output
        mistral_courses = set()
        csv_reader = csv.reader(StringIO(mistral_output))
        next(csv_reader)  # Skip headers if they exist
        for row in csv_reader:
            if row:  # Make sure row is not empty
                mistral_courses.add(row[0].strip())  # Assume course code is in the first column

        # Step 2: Split the original text by newline and filter
        original_lines = original_text.strip().split('\n')
        filtered_lines = []

        for line in original_lines:
            if not any(course_code in line for course_code in mistral_courses):
                filtered_lines.append(line)

        # Step 3: Join the filtered lines back into a single string
        filtered_text = '\n'.join(filtered_lines)
        return filtered_text
    
    def check_missing_courses(self, headers, filtered_text):
        prompt2 = f'''Below is OCR text from a student transcript. This text might contain student grade data. Determine if there is course information and corresponding grades from this data. If there is, select lines only relevant to student courses and grades and format the fields into a table in csv format. Some extracted table headers are given below to help with formatting. The csv you output should only have 3 columns: 'Course Code', 'Grade', and 'Credits', you must select which columns best fit these fields. The data could be one line long, several lines long, or if you determine that there is no grade data, simply respond with "None".

### Headers:
{headers}

### Text:
{filtered_text}

### CSV:
'''

        system_message2 = "You are a table creation assistant"
        prompt_template2=f'''<|im_start|>system
        {system_message2}<|im_end|>
        <|im_start|>user
        {prompt2}<|im_end|>
        <|im_start|>assistant
        '''

        max_tokens2 = 128
        temperature = 0
        echo = False
        stop = ["</s>"]

        print("Asking Mistral to Check it's Work...")
        try:
            # Define the parameters
            with SuppressOutput():
                model_output2 = self.llm(
                    prompt_template2,
                    max_tokens=max_tokens2,
                    temperature=temperature,
                    echo=echo,
                    stop=stop,
                )
            print("Check Complete.")
            output_string = model_output2["choices"][0]["text"].strip()
            return output_string
        
        except Exception as e:
            print(f"An error occurred during model inference: {e}")

    def string_to_dataframe(self, data_str):
        # Use StringIO to simulate a file-like object for the pandas read_csv function.
        data_io = StringIO(data_str)
        
        # Read the data into a DataFrame.
        df = pd.read_csv(data_io)

        # Strip whitespace from strings in the DataFrame
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Apply strip to column titles
        df.columns = df.columns.str.strip()
        
        return df
    
    def fix_ocr_pluses(self, dataframe, grade_header):
        # Define the mapping for grades
        map_plus = {'At': 'A+', 'Bt': 'B+', 'Ct': 'C+', 'Dt': 'D+'}

        # Check if the specified grade_header is in the dataframe's columns
        if grade_header in dataframe.columns:
            # Apply the mapping to the specified grade column
            dataframe[grade_header] = dataframe[grade_header].replace(map_plus)
        else:
            raise ValueError(f"The specified grade header '{grade_header}' is not found in the dataframe.")

        # Return the modified dataframe
        return dataframe
    
    def save_output(self, df, output_filename):
        df.to_csv(output_filename, index=False)
        print(f"DataFrame saved to {output_filename}")
    
    def process_transcript(self, headers, text, save=False, output_filename=None):
        # Query Mistral with initial text
        initial_csv_output = self.query_mistral(headers, text)
        print("Initial_CSV_Output:\n", initial_csv_output)
        initial_df = self.string_to_dataframe(initial_csv_output)

        # Filter out identified course lines from the original text
        remaining_text = self.filter_unidentified_courses(initial_csv_output, text)

        # Check for missing courses
        filtered_text = self.check_missing_courses(headers, remaining_text)
        if filtered_text.lower() != "none":
            additional_df = self.string_to_dataframe(filtered_text)
            # Append new rows to the initial DataFrame
            final_df = pd.concat([initial_df, additional_df], ignore_index=True)
        else:
            final_df = initial_df

        # Fix OCR issues with grade pluses
        final_df = self.fix_ocr_pluses(final_df, "Grade")

        # Optionally save the DataFrame
        if save:
            if output_filename:
                self.save_output(final_df, output_filename)
            else:
                print("Output filename not provided, cannot save the DataFrame.")

        return final_df

if __name__ == "__main__":

    # Sample text from Queens Unofficial Transcript
    text = """
    APSC 221 Economic And Business Practice 3.00 A 12.0
    ELEC 210 Intro Elec Circuits & Machines 4.00 A- 14.8
    MINE 244 Underground Mining 3.00 Bt 9.9
    MINE 267 App Chem/Instrument Meth Mine 4.00 A- 14.8
    MTHE 272 Applications Numerical Methods. 3.50 At 15.0
    MTHE 367 Engineering Data Analysis 4.00 B- 10.8
    APSC 100A Engineering Practice 0.00 NG 0.0
    APSC 111 Mechanics 3.50 A 14.0
    APSC 131 Chemistry And Materials 3.50 A 14.0
    APSC 151 Earth Systems And Engineering 4.00 At 17.2
    APSC 161 Engineering Graphics 3.50 A 14.0
    APSC 171 Calculus | 3.50 At 15.0
    APSC 200 Engr Design & Practice 4.00 At 17.2
    APSC 293 Engineering Communications 1.00 At 4.
    CIVL 230 Solid Mechanics | 4.25 At 18.
    MECH 230 Applied Thermodynamics | 3.50 At 15.
    MINE 201 Intro To Mining/Mineral Proces 4.00 A 16.(
    MINE 202 Comp Apps/Instrumntn In Mining 1.50 A 6.(
    MTHE 225 Ordinary Differential Equation 3.50 A 14.(
    APSC  100B Engineering Practice 11.00 A- 40.7
    APSC 112 Electricity And Magnetism 3.50 B+ 11.6
    APSC 132 Chemistry And The Environment 3.50 B 10.5
    APSC 142 Intro Computer Program Engrs 3.00 A- 11.1
    APSC 172 Calculus II 3.50 A- 13.0
    APSC 174 Introduction To Linear Algebra 3.50 At 15.0
    CLST 201 Roman History 3.00 At 12.¢
    ECON 111 Introductory Microeconomics 3.00 A- 11.1
    MINE 321 Drilling & Blasting 4.50 A- 16.6
    MINE 331 Methods Of Mineral Separation 4.50 A- 16.€
    MINE 339 Mine Ventilation 4.50 Ct 10.4
    MINE 341 Open Pit Mining 4.50 A- 16.6
    Academic Program History
    06/12/2012: Bachelor of Science Engineer Active in Program
    Major in General Engineering
    02/28/2013: Bachelor of Science Engineer Active in Program
    Major in Mining Engineering
    Option in Mining
    12/09/2014: Bachelor of Arts Active in Program
    Term GPA 3.51. Term Totals 24.00 24.00 84.3
    Term GPA 3.60 Term Totals 21.50 21.50 774
    Term GPA 4.18 Term Totals 21.75 21.75 90.8
    Term GPA 3.64 Term Totals 28.00 28.00 101.8
    Term GPA 4.13 Term Totals 18.00 18.00 74.2
    """
    # Sample Headers from Queens Unofficial Transcript
    headers = """
    Course Description Units Grade _— Points
    Course Description Units Grade Points
    Course Description Units Grade Points
    Course Description Units Grade _— Points
    Course Description Units Grade Points
    """
    data = """Course Code,Grade,Credits
    APSC 293,A+,1.00"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize Pipeline
    mistral_pipeline = MistralInference(device = device)
    # Run inference
    # table = mistral_pipeline.process_transcript(headers, text)
    tf = mistral_pipeline.string_to_dataframe(data)
    table = mistral_pipeline.fix_ocr_pluses(tf, "Grade")


    pd.set_option('display.max_rows', 40)
    print(table.head())
