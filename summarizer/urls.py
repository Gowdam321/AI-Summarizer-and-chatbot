from django.urls import path
from .views import home, summarize_text, chatbot

urlpatterns = [
    path('', home, name='home'),
    path('summarizer/', summarize_text, name='summarizer'),
    path('chatbot/', chatbot, name='chatbot'),
]
