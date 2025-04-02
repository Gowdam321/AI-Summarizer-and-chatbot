from django.http import JsonResponse
from django.shortcuts import render
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline
from django.views.decorators.csrf import csrf_exempt
import json

# Load QA and summarization models
qa_pipeline = pipeline("question-answering", model="facebook/bart-large-cnn")
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def abstractive_summary(text):
    """Generate an abstractive summary using BART"""
    summary = bart_summarizer(text, max_length=200, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def extractive_summary(text):
    """Extractive summarization using Sumy (LSA)"""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Extract 3 sentences
    return " ".join(str(sentence) for sentence in summary)

def home(request):
    """Render the home page"""
    return render(request, "home.html")

def summarizer_view(request):
    """Handles text summarization in the UI"""
    summary = None

    if request.method == "POST":
        text = request.POST.get("text", "")

        if text:
            extractive_summary_text = extractive_summary(text)
            abstractive_summary_text = abstractive_summary(text)

            # Combine summaries or choose one
            summary = abstractive_summary_text  # You can use extractive_summary_text too

            # Store summary in session for chatbot access
            request.session["summary"] = summary

    return render(request, "summarizer.html", {"summary": summary})

def chatbot(request):
    """Handles chatbot interactions"""
    if request.method == "POST":
        user_question = request.POST.get("message", "")
        summarized_text = request.session.get("summary", "")  # Retrieve summary from session

        if not user_question or not summarized_text:
            return JsonResponse({"response": "Please provide both a question and a summary."})

        response = qa_pipeline(question=user_question, context=summarized_text)
        return JsonResponse({"response": response["answer"]})  # Return chatbot response as JSON

    return render(request, "chatbot.html")  # Load chatbot UI for GET requests

@csrf_exempt
@csrf_exempt
def summarize_text(request):
    """Handles API-based summarization"""
    if request.method == "POST":
        try:
            if request.content_type == "application/json":
                data = json.loads(request.body.decode("utf-8"))
                text = data.get("text", "")
            else:
                text = request.POST.get("text", "")

            if not text:
                return JsonResponse({"error": "No text provided"}, status=400)

            extractive_summary_text = extractive_summary(text)
            abstractive_summary_text = abstractive_summary(text)

            # Choose one or combine them
            final_summary = abstractive_summary_text  # or extractive_summary_text

            return JsonResponse({
                "summary": final_summary  # Ensure this key matches frontend expectations
            }, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
    
    return render(request, "summarizer.html")


def summarizer_page(request):
    """Render the summarizer page"""
    return render(request, "summarizer.html")

def summarize_page(request):
    """Render the index.html page"""
    return render(request, "index.html")
