# api/urls.py
from django.urls import path
from .views import EmotionDetectionView

urlpatterns = [
    path("detect-emotion/", EmotionDetectionView.as_view(), name="detect-emotion"),
]
