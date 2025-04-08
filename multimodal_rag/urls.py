from django.urls import path
from .views import multimodal_search_view, index

urlpatterns = [
    path('', index, name='index'),
    path("search/", multimodal_search_view, name="multimodal_search"),
]
