from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from collections import Counter
from io import StringIO
import re
import torch
import gc

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

    def initialize(self, device):
        self.headers = ['Course Code', 'Grade', 'Credits']
        self.model_name = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"
        if device == torch.device('cuda') and not torch.cuda.is_available():
            print("Switching device to CPU.")
            device = torch.device('cpu')

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Loading Mistral to {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device,
            trust_remote_code=False,
            revision="gptq-4bit-32g-actorder_True"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    
    # Function to get a response from mistral given an OCR text string
    def query_mistral(self, text, headers, do_sample = False, temperature=0.3, top_p=0.9, top_k=40, repetition_penalty=1.1):

        # self.prompt = f'''Below is OCR text from a student transcript. This text contains a table, or multiple tables. I want you to select data only relevant to student courses and grades from these tables and format the fields into a table in csv format. Some extracted table headers are given below to help with formatting.
        
        # ### Headers:
        # {headers}

        # ### Text:
        # {text}

        # ### CSV:

        # '''

        # system_message = "You are a table creation assistant"
        # prompt_template=f'''<|im_start|>system
        # {system_message}<|im_end|>
        # <|im_start|>user
        # {self.prompt}<|im_end|>
        # <|im_start|>assistant
        # '''

        # Experimenting with having the llm put the tables into a specific format
        self.prompt = f'''Below is OCR text from a student transcript. This text contains a table, or multiple tables. Select data only relevant to student courses and grades from these tables and format the fields into a table in csv format. Some extracted table headers are given below to help with formatting. The csv you output should only have 3 columns: {self.headers[0]}, {self.headers[1]}, and {self.headers[2]}, you must select which columns best fit these fields.
        
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
        {self.prompt}<|im_end|>
        <|im_start|>assistant
        '''

        # if do sample is set to False (defualt), take only most probable next token, making the model deterministic.
        if do_sample == False:
            temperature=None
            top_p=None
            top_k=None
        
        # Create pipeline object
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=2048,
            do_sample= do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        LLM_response = pipe(prompt_template)[0]['generated_text']
        return LLM_response
    
    def extract_after_keyword(self, text, keyword = "<|im_start|>assistant"):
        """
        Extracts and returns the part of the text that comes after the first occurrence of the specified keyword.
        If the keyword is not found, the function returns an empty string.
        """
        # Find the position of the keyword in the text
        keyword_pos = text.find(keyword)

        # If the keyword is found, extract the part after the keyword
        if keyword_pos != -1:
            return text[keyword_pos + len(keyword):].strip()

        # If the keyword is not found, return an empty string
        print(f"Keyword '{keyword}' not found in text:\n{text}")
        return ""

    def extract_csv_content_flexible(self, text):
        # Split the entire text into parts separated by at least one empty line
        parts = re.split(r'\n\s*\n', text, maxsplit=1)

        # Assume the second part (if exists) contains the CSV data
        if len(parts) > 1:
            csv_part = parts[1]
        else:
            # If there's no clear separation, consider the whole text
            csv_part = parts[0]

        # Capture lines that contain at least one comma
        csv_lines = re.findall(r'^[^,]+,[^,\n]+(?:,[^,\n]+)*$', csv_part, re.MULTILINE)

        # Join the captured lines to form the CSV content
        csv_content = '\n'.join(csv_lines)

        return csv_content

    def _adjust_row_fields(self, row, target_fields):
        """
        Adjust a row to have a specific number of fields, either by merging excess fields
        or padding with None for missing fields.
        """
        if len(row) > target_fields:
            # Merge excess fields; this strategy may need customization
            row = row[:target_fields-1] + ['; '.join(row[target_fields-1:])]
        elif len(row) < target_fields:
            # Pad row with None values
            row += [None] * (target_fields - len(row))
        return row

    def preprocess_csv_data(self, csv_data):
        # Determine the most common number of fields in rows
        rows = [row.split(',') for row in csv_data.split('\n') if row]

        field_counts = Counter(len(row) for row in rows)
        common_field_count = field_counts.most_common(1)[0][0]

        # Adjust rows to have uniform number of fields
        adjusted_rows = [self._adjust_row_fields(row, common_field_count) for row in rows]

        # Convert adjusted rows to CSV string
        adjusted_csv_str = '\n'.join(','.join(str(field) for field in row) for row in adjusted_rows)
        print(adjusted_csv_str)

        # Load into DataFrame
        try:
            df = pd.read_csv(StringIO(adjusted_csv_str))
        except Exception as e:
            print("failed to read String.")

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
    
    # Function to process everything in sequence
    def get_table(self, table_and_header_dict, output_directory = None):
        
        dataframes = []

        for image_name in table_and_header_dict:
            # Grab strings from dictionary
            header_string = table_and_header_dict[image_name]['header_data']
            table_string = table_and_header_dict[image_name]['table_data']
            
            # Ask mistral
            LLM_response = self.query_mistral(table_string, header_string)
            # Adjusted to pass "assistant" keyword
            LLM_response_filtered = self.extract_after_keyword(LLM_response) 

            try:
                # Flexible extraction to handle varying number of fields in CSV data
                csv_data = self.extract_csv_content_flexible(LLM_response_filtered)
            except Exception as e:
                print(f"Flexible extraction failed due to {e}, passing filtered response.")
                csv_data = LLM_response_filtered

            # Process and load the CSV data into a DataFrame
            try:
                df = self.preprocess_csv_data(csv_data.replace('"', ''))  # Using the new preprocessing function
            except Exception as e:
                print(f"Error processing CSV data: {e}")
            
            df_cleaned = self.fix_ocr_pluses(df, self.headers[1])
        
            if output_directory is not None:
                self.save_output(df_cleaned, output_directory + image_name)
            
            dataframes.append(df_cleaned)
            
        return dataframes