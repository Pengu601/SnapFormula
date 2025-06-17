from django.urls import path
from .views import ImageToTextView

urlpatterns=[
    path('upload/', ImageToTextView.as_view(), name ='image-to-latex')
    ]

