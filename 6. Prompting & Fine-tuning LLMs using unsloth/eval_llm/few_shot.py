from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import re

import pandas as pd

def remove_punctuation(text):
    # Utilise une expression régulière pour remplacer toutes les ponctuations par une chaîne vide
    return re.sub(r'[^\w\s]', '', text)

def few_shot_call(question):

    # Define the prompt
    few_shot_prompt = ChatPromptTemplate.from_template("""
    Context: You are an AI assistant with access to the logs of an ERP system for a large company.
    The company has three main departments: Sales, Maintenance, and Inventory Management.
    The goal is to provide precise and quick answers to queries from each department.

    Example 1:
    Question: "What is the total revenue for the last quarter?"
    Response: "Sales"

    Example 2:
    Question: "How many sales were made this month?"
    Response: "Sales"

    Example 3:
    Question: "What are the top 5 products sold this year?"
    Response: "Sales"

    Example 4:
    Question: "What are the most common types of failures?"
    Response: "Maintenance"

    Example 5:
    Question: "What corrective actions were taken for issue X?"
    Response: "Maintenance"

    Example 6:
    Question: "List the maintenance activities performed last week."
    Response: "Maintenance"

    Example 7:
    Question: "What is the current stock level of item Y?"
    Response: "Inventory Management"

    Example 8:
    Question: "How many units of product Z are in stock?"
    Response: "Inventory Management"

    Example 9:
    Question: "Which items are low in stock and need reordering?"
    Response: "Inventory Management"

    Task: Based on the examples above, answer the following question using the ERP logs:

    Question: {question}

    Response: Provide only the class label for the question. The class labels are as follows:
    - Sales
    - Maintenance
    - Inventory Management
    """)

    # Liste des modèles
    models = ['llama3:latest', 'mistral:latest', 'phi3:latest']

    # Initialiser un DataFrame vide avec un MultiIndex pour les colonnes
    columns = pd.MultiIndex.from_product([['FEW_SHOT'], models], names=['Type', 'Model'])
    df = pd.DataFrame(columns=columns)
    
    # Préparer une liste pour stocker les réponses
    responses = []

    for model in models:
        # Load the model
        llm = ChatOllama(model=model)
        
        # create the zero-shot chain
        few_shot_chain = few_shot_prompt | llm | StrOutputParser()

        # Run the zero-shot chain
        response = few_shot_chain.invoke(input=question)

        # Nettoyer la réponse
        if 'sales' in response.lower():
            response = 'Sales'

        elif 'maintenance' in response.lower():
            response = 'Maintenance'

        elif 'inventory Management' in response.lower():
            response = 'Inventory Management'
        
        # Ajouter la réponse à la liste des réponses
        responses.append(response)

    # Ajouter les réponses au DataFrame
    df.loc[0] = responses
    
    return df