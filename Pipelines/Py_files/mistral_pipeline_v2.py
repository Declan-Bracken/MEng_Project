from llama_cpp import Llama
import pandas as pd
from io import StringIO
import torch
import gc
import os
import sys
import csv
import re

# For output suppression of the LLM logging
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
    def __new__(cls, device=torch.device('cuda'), model_path=None, new_instance = False, stream = False):
        if cls._instance is None or new_instance == True:
            print("Creating new instance of MistralInference")
            cls._instance = super(MistralInference, cls).__new__(cls)

            # Initialize instance variables
            cls._instance.initialize(device, model_path)
            cls.stream = stream
        else:
            print("Using existing instance of MistralInference")

        return cls._instance
    
    def initialize(self, device, model_path, max_sequence_len = 4096):
        if not model_path:
            raise ValueError("Model path must be provided.")
        
        # Output Collumns
        # self.headers = ['Course Code', 'Grade', 'Credits']

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
            
    def query_mistral(self, text, headers = None):
#         self.prompt_templates = [f'''I will provide you with OCR text of a grade table from a student transcript. Format the fields into a table in csv format. The csv you output should only have the following columns: '{self.headers[0]}' or 'Course Title' (or both), '{self.headers[1]}', and '{self.headers[2]}'. you must select which columns best fit these fields. For example, this is how you would format the following table:
# Below is OCR text of a grade table from a student transcript. Format the fields into a table in csv format. The csv you output should only have 3 columns: '{self.headers[0]}' or 'Course Title' (or both), '{self.headers[1]}', and '{self.headers[2]}'. you must select which columns best fit these fields. For example, this is how you would format the following table:

        self.prompt_templates = [
f'''Below is OCR text from a student transcript. This text contains a table, or multiple tables. Select data only relevant to student courses and grades from these tables and format the fields into a table in csv format. Please output only the CSV table and no other preceding or trailing text in your response.
                                                        
### Text:
{headers}
{text}''',
f'''Below is OCR text from a student transcript. This text contains a table, or multiple tables. Select data only relevant to student courses and grades from these tables and format the fields into a table in csv format. **Output only the CSV table**
### Text:
{text}
''']

        if headers is not None:
            prompt = self.prompt_templates[0]
        else:
            prompt = self.prompt_templates[1]

        system_message = "You are a helpful assistant." #You take input input data and clean/organize it into CSV texts
        prompt_template=f'''<|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant'''

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
                    top_p=0,
                    echo=echo,
                    stop=stop,
                    stream = self.stream)
            print("Prompting Successful!")
            # Handle streamed or non-streamed output
            if self.stream:
                print("Streaming Inference Now...")
                final_result = ""
                for chunk in model_output:
                    token = chunk["choices"][0]["text"]
                    final_result += token
                output_text = final_result
            else:
                output_text = model_output["choices"][0]["text"].strip()
                
            # Extract CSV using the new method
            csv_data = self.extract_csv(output_text)
            return csv_data, model_output
        except Exception as e:
            print(f"An error occurred during model inference: {e}")
            return None, None

    def extract_csv(self, output_text):
        """
        Extracts the CSV part from the output text using more robust logic.
        """
        # Use regex to find the part of the text that resembles a CSV table
        csv_match = re.search(r"(?<=\n)([^\n,]+,)+[^\n,]+", output_text)
        if csv_match:
            csv_text = csv_match.group(0)
            return csv_text
        else:
            print("No valid CSV found in the output.")
            return ""
        
    def string_to_dataframe(self, data_str):
        """
        Converts a string in CSV format to a pandas DataFrame.
        """
        try:
            # Use StringIO to simulate a file-like object for the pandas read_csv function.
            data_io = StringIO(data_str)
            # Read the data into a DataFrame.
            df = pd.read_csv(data_io)
            # Strip whitespace from strings in the DataFrame
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            # Apply strip to column titles
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            print(f"Failed converting string to dataframe: {e}")
            return pd.DataFrame()
    
    def clean_data(self, dataframe):
        # Remove unwanted characters from all cells
        dataframe = dataframe.replace({r'[;:"()$%^#*@!]': ''}, regex=True)
        # Clean the column headers
        dataframe.columns = dataframe.columns.str.replace(r'[;:"()$%^#*@!]', '', regex=True)

        # Strip leading/trailing whitespaces from headers and all cells
        dataframe.columns = dataframe.columns.str.strip()
        dataframe = dataframe.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Define the mapping for grades
        # map_plus = {'At': 'A+', 'Bt': 'B+', 'Ct': 'C+', 'Dt': 'D+'}
        # Check if the specified grade_header is in the dataframe's columns
        # if grade_header in dataframe.columns:
        #     # Apply the mapping to the specified grade column
        #     dataframe[grade_header] = dataframe[grade_header].replace(map_plus)
        # else:
        #     print(f"The specified grade header '{grade_header}' is not found in the dataframe.")
        # Return the modified dataframe
        return dataframe
    
    def save_output(self, df, output_filename):
        df.to_csv(output_filename, index=False)
        print(f"DataFrame saved to {output_filename}")
    
    def process_transcript(self, headers, text, save=False, output_filename=None):
        # Query Mistral with initial text
        self.csv_output, self.model_output = self.query_mistral(text, headers=headers)
        print(self.csv_output)
        # Convert to Dataframe
        self.df = self.string_to_dataframe(self.csv_output)
        # Fix OCR issues with grade pluses
        self.final_df = self.clean_data(self.df) # Second header is for grades
        # Optionally save the DataFrame
        if save:
            if output_filename:
                self.save_output(self.final_df, output_filename)
            else:
                print("Output filename not provided, cannot save the DataFrame.")

        return self.final_df

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
    table.head(40)
