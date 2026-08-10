AI Study Assistant Agent
Lucas Montanaro

Main files

1. AI_Study_Assistant_Agent.ipynb
This is the complete Google Colab notebook. It downloads SciQ from Kaggle, cleans the data, trains the answer choice model, builds the search index, evaluates the agent, and opens the chat interface.

2. ai_study_assistant.py
This is the same project as a normal Python program.

3. AI_Study_Assistant_Final_Report.docx
This is the full final report.

How to run in Google Colab

1. Open Google Colab.
2. Upload AI_Study_Assistant_Agent.ipynb.
3. Select Runtime, then Run all.
4. Allow the first cell to install the required libraries.
5. The notebook will download the public SciQ dataset from Kaggle.
6. Read the printed evaluation results.
7. Open the Gradio link produced by the final interface cell.

The project does not need a paid AI API key. If Kaggle asks for an account token, open Kaggle account settings, create an API token, upload kaggle.json to Colab, and run the data cell again.
