from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd

def zero_shot_method(question):

    # Define the prompt
    zero_shot_prompt = ChatPromptTemplate.from_template("""Context: You are an AI assistant with access to the logs of an ERP system for a large company.
                                The company has three main departments: Sales, Maintenance, and Inventory Management.
                                The goal is to provide precise and quick answers to queries from each department.\n

                                Task: Answer the following question using the ERP logs:\n
                                
                                Question: {question}\n

                                Response: IMPORTANT!! Provide only the class label for the question. The class labels are as follows:
                                        Sales
                                        Maintenance
                                        Inventory Management\n""")
    # Liste des modèles
    models = ['llama3:latest', 'mistral:latest', 'phi3:latest']

    # Initialiser un DataFrame vide avec un MultiIndex pour les colonnes
    columns = pd.MultiIndex.from_product([['ZERO_SHOT'], models], names=['Type', 'Model'])
    df = pd.DataFrame(columns=columns)
    
    # Préparer une liste pour stocker les réponses
    responses = []

    for model in models:
    
        # Load the model
        llm = ChatOllama(model=model)
        
        # create the zero-shot chain
        zero_shot_chain  = zero_shot_prompt | llm | StrOutputParser()

        # Run the zero-shot chain
        response = zero_shot_chain.invoke(input=question)
        
        # Nettoyer la réponse
        if 'sales' in response.lower():
            response = 'Sales'

        elif 'maintenance' in response.lower():
            response = 'Maintenance'

        elif 'inventory management' in response.lower():
            response = 'Inventory Management'

        # Ajouter la réponse à la liste des réponses
        responses.append(response)

    # Ajouter les réponses au DataFrame
    df.loc[0] = responses
    
    return df