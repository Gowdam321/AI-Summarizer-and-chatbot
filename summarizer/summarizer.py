from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the Pegasus model and tokenizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

def summarize_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load the T5 model and tokenizer for the chatbot
tokenizer_chatbot = T5Tokenizer.from_pretrained("t5-small")
model_chatbot = T5ForConditionalGeneration.from_pretrained("t5-small")

def chatbot_response(question, context=""):
    input_text = f"question: {question} context: {context}"
    
    inputs = tokenizer_chatbot(input_text, return_tensors="pt", max_length=512, truncation=True)
    response_ids = model_chatbot.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
    response = tokenizer_chatbot.decode(response_ids[0], skip_special_tokens=True)
    
    return response
