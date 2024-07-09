import ollama
import gradio as gr

def generator_yoda(prompt):

    response = ollama.generate(model='yoda', prompt=prompt)
    response = response['response']
    return response

iface = gr.Interface(
    fn=generator_yoda,
    inputs=gr.Textbox(lines=7, label="Question to Yoda"),
    outputs=gr.Textbox(label="... Yoda Response ..."),
    title="Yoda Talk Generator",
    description="Generate text in the style of Yoda."
)

iface.launch()