from django.urls import path
from .views import search_view, index

urlpatterns = [
    path('', index, name='index'),  # Serve frontend at "/"
    path('search/', search_view, name='search'),  # API search endpoint
]