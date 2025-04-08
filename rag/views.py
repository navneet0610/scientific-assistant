from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .faiss_search import search_faiss_index
from .utils import extract_text_from_audio, extract_text_from_image

def index(request):
    return render(request, 'rag/index.html')

@csrf_exempt
def search_view_audio(request):
    """Handles audio file uploads, extracts text using Whisper, and performs FAISS search."""
    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]
        transcribed_text = extract_text_from_audio(audio_file)

        if not transcribed_text:
            return JsonResponse({"error": "Failed to extract text from audio"}, status=400)

        # Perform FAISS search with transcribed text
        results = search_faiss_index(transcribed_text, top_k=5)
        return JsonResponse({"results": results})

    return JsonResponse({"error": "No audio provided"}, status=400)


@csrf_exempt
def search_view_image(request):
    """Handles image uploads, extracts text using OCR, and performs FAISS search."""
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]
        extracted_text = extract_text_from_image(image_file)

        if not extracted_text:
            return JsonResponse({"error": "Failed to extract text from image"}, status=400)

        # Perform FAISS search with extracted text
        results = search_faiss_index(extracted_text, top_k=5)
        return JsonResponse({"results": results})

    return JsonResponse({"error": "No image provided"}, status=400)


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
