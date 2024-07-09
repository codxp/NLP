from re import S
import re
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
import pandas as pd

from pydantic import Json
from regex import F

load_dotenv()

count_error = 0

def preprocess_text(question: str) -> list:
    
    try:
        question = question.replace('\n', '')
        question = question.replace('  ', '')
        question = question.replace('Sales:', '[[')
        question = question.replace('Maintenance:', '],[')
        question = question.replace('Inventory Management:', '],[')
        question = f"{question}]]"
        list_string = eval(question)

        return list_string
    
    except Exception as e:
        count_error += 1
        print('Erreur: ', e)
        return '[{erreur: Erreur lors de la conversion de la question en liste}]'


llm = AzureChatOpenAI(
    openai_api_version='2024-02-15-preview',
    azure_deployment='flo-gpt4',
    temperature=0.9)


generation_prompt = """
I want you to act as an AI assistant that generates synthetic questions for an ERP system used by a large company. The company has three main departments: Sales, Maintenance, and Inventory Management. Keep in mind previous generated questions and try to generate diverse questions for each department.

Examples:

Sales:
1. 
    {
        "input": "What is the total revenue for the last quarter?",
        "category": "Sales"
    }
2. 
    {
        "input": "How many sales were made this month?",
        "category": "Sales"
    }
3. 
    {
        "input": "What are the top 5 products sold this year?",
        "category": "Sales"
    }

Maintenance:
1. 
    {
        "input": "What are the most common types of failures?",
        "category": "Maintenance"
    }
2. 
    {
        "input": "What corrective actions were taken for issue X?",
        "category": "Maintenance"
    }
3. 
    {
        "input": "List the maintenance activities performed last week.",
        "category": "Maintenance"
    }

Inventory Management:
1. 
    {
        "input": "What is the current stock level of item Y?",
        "category": "Inventory Management"
    }
2. 
    {
        "input": "How many units of product Z are in stock?",
        "category": "Inventory Management"
    }
3. 
    {
        "input": "Which items are low in stock and need reordering?",
        "category": "Inventory Management"
    }

Task: Based on the examples above, generate one additional questions for each department. response will be UNIQUELY a JSON object with the following structure:
    {Departement name: {
        "input": "Generated question",
        "category": "Department name"
                        }
    
    }
"""

result = []

for _ in range(120):
    try:
        print(f'Iteration: {_}')
        response = llm.invoke(input=generation_prompt)
        question = response.content
        list_string = preprocess_text(question)

        for i in list_string:

            for j in i:

                result.append(j)
    except Exception as e:
        print(f'Une erreur est survenue lors de l itération {_}: {e}')
        continue


df = pd.DataFrame(result)
df.to_csv('generated_questions.csv', index=False)

print('Data saved successfully!')
