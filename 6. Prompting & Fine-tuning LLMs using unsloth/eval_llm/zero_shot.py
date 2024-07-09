from unittest import result
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def zero_shot(question):

    models = ['llama3:latest', 'mistral:latest', 'phi3:latest']

    # Define the prompt
    zero_shot_prompt = ChatPromptTemplate.from_template("""Context: You are an AI assistant with access to the logs of an ERP system for a large company.
                                The company has three main departments: Sales, Maintenance, and Inventory Management.
                                The goal is to provide precise and quick answers to queries from each department.\n

                                Task: Answer the following question using the ERP logs:\n
                                
                                Question: {question}\n

                                Response: Provide only the class label for the question. The class labels are as follows:
                                        - Sales
                                        - Maintenance
                                        - Inventory Management\n""")

    # Define the question
    # questions = ["What is the total number of sales transactions in the last month?",
    #             "What is the average time taken to resolve maintenance requests?",
    #             "What is the total number of items in the inventory?"]

    result = {}

    for model in models:

        result['Zero_shot'] = {}

        
        
        # Load the model
        llm = ChatOllama(model=model)
        
        # create the zero-shot chain
        zero_shot_chain  = zero_shot_prompt | llm | StrOutputParser()

        # Run the zero-shot chain
        response = zero_shot_chain.invoke(input=question)
        result['Zero_shot'][model] = response

    return result