from transformers import pipeline, BertTokenizer, RobertaTokenizer
import gradio as gr

# Charger les modèles et tokenizers
model_bert = pipeline("text-classification", model="FloDevIA/results_bert", tokenizer="FloDevIA/bert-tokenizer-projetnlp")
model_roberta = pipeline("text-classification", model="FloDevIA/results_roberta", tokenizer="FloDevIA/roberta-tokenizer-projetnlp")

# Correspondance entre les labels numériques et les sujets
label_to_subject = {
    'LABEL_0': 'biology',
    'LABEL_1': 'chemistry',
    'LABEL_2': 'computer',
    'LABEL_3': 'maths',
    'LABEL_4': 'physics',
    'LABEL_5': 'social sciences'
}

def predict_subject(text, model_choice):
    try:
        # inputs = preprocess_text(text, model_choice)
        if model_choice == "Model 1 (BERT)":
            predictions = model_bert(text)
        elif model_choice == "Model 2 (RoBERTa)":
            predictions = model_roberta(text)
        else:
            return "Model choice is not recognized."

        if predictions:
            label = predictions[0]['label']
            if label in label_to_subject:
                predictions[0]['label'] = label_to_subject[label]
            return f"{predictions[0]['label']}, Score: {predictions[0]['score']:.4f}"
        else:
            return "No predictions returned by the model."

    except Exception as e:
        # This catches any error and returns it directly
        return f"Error processing input: {e}"

def documentation():
    return """
    ## Subject Classification Documentation
    Welcome to the Subject Classification Demo! This interactive application leverages advanced machine learning models to classify the subject of the text you provide.

    ### About the Technology
    Subject classification involves categorizing text into predefined topics. The underlying models are trained on large datasets to learn how to predict the subject of a text accurately.
    This demo utilizes two different models from the Hugging Face Transformers library:
    
    - **Model 1**: BERT for subject classification fine-tuned for various English text sources.
        - Pretrained model: `bert-base-uncased`
        - Dataset used for fine-tuning : https://huggingface.co/datasets/vishalp23/subject-classification
        - 6 subjects: biology, chemistry, computer, maths, physics, social sciences.
        - Training metrics:

                    | Loss   | Epoch | Step  | Validation Loss | Validation Accuracy |
                    |--------|-------|-------|-----------------|---------------------|
                    | 0.1069 | 1.0   | 5405  | 0.1083          | 0.9733              |
                    | 0.0474 | 2.0   | 10810 | 0.0881          | 0.9810              |
                    | 0.0162 | 3.0   | 16215 | 0.0828          | 0.9847              |


    - **Model 2**: RoBERTa for subject classification fine-tuned similarly for diverse English texts.
        - Pretrained model: `roberta-base`
        - Dataset used for fine-tuning : https://huggingface.co/datasets/vishalp23/subject-classification
        - 6 subjects: biology, chemistry, computer, maths, physics, social sciences.
        - Training metrics:

                    | Loss   | Epoch | Step  | Validation Loss | Validation Accuracy |
                    |--------|-------|-------|-----------------|---------------------|
                    | 0.1399 | 1.0   | 5405  | 0.1215          | 0.9696              |
                    | 0.0617 | 2.0   | 10810 | 0.1011          | 0.9779              |


    ### Data Privacy
    We value your privacy. All input texts are processed in real-time and are not stored or used for any purpose other than subject classification during your session.
    Enjoy exploring the classification of subjects in text with our cutting-edge AI models! 🙂
    """

with gr.Blocks(title="Subject Classification", theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        with gr.TabItem("Demo"):
            gr.Markdown("### Subject Classification Demo\nEnter your text and select a model to get the subject classification.")
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(label="Input Text", placeholder="Type here or select an example...")
                    model_choice = gr.Radio(["Model 1 (BERT)", "Model 2 (RoBERTa)"], label="Model Choice", value="Model 1 (BERT)")
                    submit_button = gr.Button("Analyze")
                with gr.Column():
                    output = gr.Label()

            examples = gr.Examples(examples=[
                "The cell is the basic structural and functional unit of all known living organisms.",
                "Chemical reactions are processes that lead to the transformation of one set of chemical substances to another.",
                "Artificial intelligence is a branch of computer science dealing with the simulation of intelligent behavior in computers.",
                "Calculus is the mathematical study of continuous change.",
                "Quantum mechanics is a fundamental theory in physics describing the properties of nature on an atomic scale.",
                "Sociology is the study of social behavior, society, patterns of social relationships, social interaction, and culture."
            ], inputs=text_input)

            submit_button.click(
                predict_subject, 
                inputs=[text_input, model_choice],
                outputs=output
            )
        
        with gr.TabItem("Documentation"):
            doc_text = gr.Markdown(documentation())

demo.launch()
