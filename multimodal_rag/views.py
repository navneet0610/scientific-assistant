import os
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import logging
from multimodal_rag.multimodal_faiss_search import search_image, search_text, BASE_DIR


def index(request):
    """
    Renders the main page where users can interact with the multimodal AI search.
    """
    return render(request, 'index.html')

@csrf_exempt
@require_POST
def multimodal_search_view(request):
    try:
        logger = logging.getLogger("multimodal_rag.views")
        is_image = request.POST.get("is_image", "false").lower() == "true"

        if is_image:
            if 'image' not in request.FILES:
                return JsonResponse({"error": "Image file is required for image search."}, status=400)
            logger.info("request received with image")
            image_file = request.FILES['image']
            temp_image_path = os.path.join(BASE_DIR, f"tmp/{image_file.name}")
            with open(temp_image_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            results = search_image(temp_image_path)
            os.remove(temp_image_path)

        else:
            logger.info("request received with query")
            query = request.POST.get("query", "").strip()
            if not query:
                return JsonResponse({"error": "Text query is required for text search."}, status=400)

            results = search_text(query)

        return JsonResponse(results, safe=False)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
