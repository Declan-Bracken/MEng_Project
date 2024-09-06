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
    def __new__(cls, device=torch.device('cuda'), model_path=None, new_instance = False):
        if cls._instance is None or new_instance == True:
            print("Creating new instance of MistralInference")
            cls._instance = super(MistralInference, cls).__new__(cls)

            # Initialize instance variables
            cls._instance.initialize(device, model_path)
        else:
            print("Using existing instance of MistralInference")

        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Resets the singleton instance."""
        print("Resetting the MistralInference singleton instance.")
        cls._instance = None
    
    def initialize(self, device, model_path, max_sequence_len = 3072):
        if not model_path:
            raise ValueError("Model path must be provided.")
        
        # Output Collumns
        # self.headers = ['Course Code', 'Grade', 'Credits']

        if device == torch.device('cuda') and not torch.cuda.is_available():
            print("Switching device to CPU.")
            device = torch.device('cpu')

        torch.cuda.empty_cache()
        gc.collect()

        # Dynamically determine optimal number of threads
        n_threads = max(1, torch.get_num_threads() - 1)  # Leave one thread for OS and other processes
        print(f"Using {n_threads} CPU threads.")

        # Determine the number of layers to offload to GPUs
        if device.type == 'cuda':
            available_gpus = torch.cuda.device_count()
            total_memory_per_gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Total GPUs available: {available_gpus}")
            print(f"Total GPU memory available per GPU: {total_memory_per_gpu_gb:.2f} GB")

            # Decide how many layers to offload per GPU

            n_gpu_layers = min(24, int(total_memory_per_gpu_gb // 1.2))  # Example: 1.2 GB per layer
            print(f"Offloading {n_gpu_layers} layers per GPU across {available_gpus} GPUs.")

        else:
            n_gpu_layers = 0  # No GPU offloading

        print(f"""-----------------------------------------------------------------------\nLoading Mistral to {device}...\n-----------------------------------------------------------------------""")
        # Set up LLaMA model with multi-GPU support
        with SuppressOutput():  # Suppress Verbose Output
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=max_sequence_len,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                    n_gpus=available_gpus  # Enable multiple GPUs
                )
                print("Model Loaded Successfully with multi-GPU support!")
            except Exception as e:
                print(f"Error Loading Model Weights: {e}")

    def extract_csv(self, output_text):
        """
        Extracts the CSV part from the output text using a more robust regex to capture the entire CSV table.
        
        Args:
        - output_text (str): The text output from the LLM that may contain CSV and non-CSV content.

        Returns:
        - str: The extracted CSV portion of the text if found; otherwise, an empty string.
        """
        # Use regex to find the part of the text that resembles a CSV table
        # This regex matches multiple lines that follow the CSV format
        csv_match = re.search(r"((?:[^\n,]+,)+[^\n,]+\n?)+", output_text)

        if csv_match:
            csv_text = csv_match.group(0)
            return csv_text.strip()  # Remove any trailing newline characters
        else:
            print("No valid CSV found in the output.")
            return ""

        
    def query_mistral(self, text, headers = None):
        prompt = f'''Below is OCR text from a student transcript. This text contains a table, or multiple tables. Select data only relevant to student courses and grades from these tables and format the fields into a table in csv format.
                                                                
### Text:
{headers}
{text}'''

        system_message = "You are a helpful assistant."
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

        print("Prompting Mistral...")
        print(prompt)

        # Define the parameters
        model_output = self.llm(
            prompt_template,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=echo,
            stop=stop,
            stream = True
        )

        print("Streaming Inference Now...")

        # Initialize a variable to store the full output
        final_result = ""

        # Iterate over the streamed output and print tokens as they are generated
        for chunk in model_output:
            # Access the text from the chunk
            token = chunk["choices"][0]["text"]
            
            # Append the token to the final result
            final_result += token
            
            # Print the token (or process it as needed)
            print(token, end='', flush=True)

        # After streaming is complete, you can use `final_result` as the full generated output
        # print("\nFinal Output:", final_result)
        return self.extract_csv(final_result.replace('<|im_end|>','').strip())
     

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
    csv_string = mistral_pipeline.query_mistral(text, headers=headers)

