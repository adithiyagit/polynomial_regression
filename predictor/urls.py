from django.urls import path
from .views import predict_yield

urlpatterns = [
    path('', predict_yield, name='predict_yield'),
]
