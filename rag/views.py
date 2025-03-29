from django.http import JsonResponse
from .faiss_search import search_arxiv


def search_view(request):
    query = request.GET.get("q", "")
    if not query:
        return JsonResponse({"error": "Query parameter 'q' is required"}, status=400)

    results = search_arxiv(query, top_k=5)
    return JsonResponse({"results": results})
