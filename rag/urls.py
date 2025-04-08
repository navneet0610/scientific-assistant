from django.urls import path
from .views import search_view, index, search_view_image, search_view_audio

urlpatterns = [
    path('', index, name='index'),  # Serve frontend at "/"
    path('search/', search_view, name='search'),  # Text-based search
    path('search/process_audio/', search_view_audio, name='search_audio'),  # Audio-based search
    path('search/process_image/', search_view_image, name='search_image'),  # Image-based search
]
