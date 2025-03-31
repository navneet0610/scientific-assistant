from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .faiss_search import search_faiss_index  # Ensure this matches the function name


@csrf_exempt  # Allows API testing without CSRF token
def search_view(request):
    """Django view for searching ArXiv papers using FAISS."""

    query = request.GET.get("q", "").strip()  # Get query from URL parameters

    if not query:
        return JsonResponse({"error": "Query parameter 'q' is required"}, status=400)

    try:
        results = search_faiss_index(query, top_k=5)  # Fetch top 5 papers

        if not results:
            return JsonResponse({"message": "No relevant papers found."}, status=200)

        return JsonResponse({"results": results}, status=200)

    except Exception as e:
        return JsonResponse({"error": f"Internal Server Error: {str(e)}"}, status=500)
