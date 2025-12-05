from django.urls import path
from .views import predict_diabetes, predict_heart_disease, predict_kidney_disease, predict_cancer, predict_parkinson

urlpatterns = [
    path("diabetes/", predict_diabetes),
    path("heart-disease/", predict_heart_disease),
    path("kidney-disease/", predict_kidney_disease),  # FIXED: Changed to kidney-disease
    path("cancer/", predict_cancer),
    path("parkinson/", predict_parkinson),
]