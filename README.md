The current admissions process for the University of Toronto M.Eng program requires administrative staff to manually examine and sort through applicant documents to create organized lists of applicants and their relevant data (School, Program, GPA, etc.). This process can be tedious and inefficient, especially considering the admissions documents (transcripts, cover letters, resumes, etc.) vary drastically in their information and formatting across different individuals, schools, and countries. This project aims to answer the research question; Can we automate the admissions process by setting up a pipeline to extract applicant information from their admissions documents in a reliable and secure manner? Furthermore, can this extracted information be optimized for data-driven decision making in the admissions process? This project will examine the degree of automation possible using modern data science techniques and available software resources.

The resulting pipeline model will drastically improve worker efficiency, aswell as admissions consistency between applicants by standardizing the way data is processed and viewed. The model will be deployed with a human-in-the-loop philosophy to catch edge cases and check work. This pipeline is being designed to create an organized relational database from just images, allowing for a series of impactful data-science projects in the future including:
- A dashboard to give administrative staff advanced student analytics at a glance for admissions decision-making.
- Predicting student success based prior academic performance and areas of specialization.
- Predicting the likelihood of student enrollment contingent on recieving an offer.

Version 1.0
This pipeline takes images of student transcripts, crops regions containing grade data using a custom vision model, uses Tesseract's open-source OCR engine to convert the images into machine-readable code, and then reconstructs the table by performing row-wise and column-wise clustering using the word-level bounding boxes from OCR. 

Version 2.0
Replaces fast clustering algorithms with a slower but powerful LLM (Mistral or Mixtral) for table reconstruction and interpretation.

Version 3.0
Implements a multimodal LLM (MiniCPM-Llama3-V-2_5) fine tuned on a custom-built dataset of image-text pairs to transcribe images in a single prompt at 1/100 the size of GPT4o but with similar reading and organizational ability.
