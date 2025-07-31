from django.urls import path
from .views import home, summarize_text, chatbot

urlpatterns = [
    path('', home, name='home'),  # home page
    path('api/summarizer/', summarize_text, name='summarizer'),  # POST API for summarization
    path('api/chatbot/', chatbot, name='chatbot'),  # POST API for chatbot
]
