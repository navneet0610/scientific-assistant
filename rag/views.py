from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .faiss_search import search_faiss_index

def index(request):
    return render(request, 'rag/index.html')

@csrf_exempt
def search_view(request):
    """API endpoint for FAISS similarity search."""
    query = request.GET.get("q", "").strip()

    if not query:
        return JsonResponse({"error": "Query parameter 'q' is required"}, status=400)

    try:
        results = search_faiss_index(query, top_k=5)

        if not results:
            return JsonResponse({"message": "No relevant papers found."}, status=200)

        return JsonResponse({"results": results}, status=200)

    except Exception as e:
        return JsonResponse({"error": f"Internal Server Error: {str(e)}"}, status=500)
