from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

load_dotenv()

llm = AzureChatOpenAI(
    openai_api_version='2024-02-15-preview',
    azure_deployment='flo-gpt4',
    temperature=0.9)

def generate_response(prompt, context, question, temperature):
   
    llm = AzureChatOpenAI(
    openai_api_version='2024-02-15-preview',
    azure_deployment='flo-gpt4',
    temperature=temperature)

    prompt_template = prompt + "\n\n" + context + "\n\n" + question
    # Generate the travel plan using the model
    response = llm.invoke(prompt_template)
    response = response.content
    return response

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=[gr.Dropdown(choices=["Examine the current scientific understanding.",
    "Discuss the implications and future projections.",
    "Provide a detailed analysis."], label="Prompt"),
        
        gr.Dropdown(choices=["The biodiversity on Earth is facing significant threats due to climate change. These include alterations in habitat, migration patterns, and extinction rates.",
    "Climate change not only affects temperature but also the ecosystems that support various species. These changes disrupt food chains and breeding grounds.",
    "Many species are unable to adapt quickly enough to the rapid pace of climate change, leading to a decline in biodiversity. This loss has wider implications for ecosystem services that humans rely on."], label="Context"),

        gr.Dropdown(choices=["What are the predicted long-term effects of climate change on terrestrial biodiversity?",
    "How might rising sea levels and increased temperatures influence marine ecosystems over the next fifty years?",
    "Can you describe potential strategies that could mitigate the effects of climate change on biodiversity in freshwater habitats?"], label="Question"),
    gr.Slider(minimum=0, maximum=1, step=0.01, label="Model Temperature")
    ],

    outputs=gr.Textbox(label="Response"),
    title="Climate Impact Explorer: Biodiversity and Ecosystems",
    description="Investigate the influence of prompt, context, and question on the study of biodiversity and ecosystems."
)

# Launch the interface
iface.launch()