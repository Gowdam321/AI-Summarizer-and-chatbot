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

@csrf_exempt
def chatbot(request):
    """Handles chatbot interactions via AJAX"""
    if request.method == "POST":
        question = request.POST.get("question", "")  # match frontend key
        summarized_text = request.session.get("summary", "")  # from session

        if not question or not summarized_text:
            return JsonResponse({"answer": "Please provide a question and summarize the text first."})

        result = qa_pipeline(question=question, context=summarized_text)
        return JsonResponse({"answer": result["answer"]})  # match frontend key

    return JsonResponse({"error": "Invalid request method"}, status=400)


@csrf_exempt
def summarize_text(request):
    """Handles AJAX-based summarization (POST) and renders page (GET)"""
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

            final_summary = abstractive_summary_text

            # ✅ Save summary in session
            request.session["summary"] = final_summary

            return JsonResponse({"summary": final_summary}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)

    # ✅ If GET request, render the combined summarizer + chatbot page
    return render(request, "summarizer.html")



def summarizer_page(request):
    """Render the summarizer page"""
    return render(request, "summarizer.html")

def summarize_page(request):
    """Render the index.html page"""
    return render(request, "index.html")
