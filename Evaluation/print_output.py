import json
file_path = "Evaluation\Test_Results\pipeline_results.json"
img_name = "Synthetic_Image_Annotation/Test_Data/images/1a2SqGFdFGGX81zAHb9-9r7vi2r_SnkXb.jpg"
with open(file_path,'r') as f:
    file = json.load(f)
print(file[img_name])