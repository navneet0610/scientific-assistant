import torch
import torchaudio
import io
import tempfile

import easyocr
from pydub import AudioSegment
from transformers import pipeline


def extract_text_from_audio(audio_file):
    """
    Transcribes speech from an uploaded audio blob using Whisper.

    Args:
        audio_file (InMemoryUploadedFile): Audio file sent from frontend.

    Returns:
        str: Transcribed text.
    """
    try:
        # Read uploaded audio file into BytesIO
        audio_bytes = io.BytesIO(audio_file.read())

        # Convert BytesIO to WAV using pydub
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp_audio:
            audio = AudioSegment.from_file(audio_bytes, format="wav")  # Ensure it's in WAV format
            audio.export(tmp_audio.name, format="wav")  # Save as WAV

            # Load audio with torchaudio
            waveform, sample_rate = torchaudio.load(tmp_audio.name)

        # Convert to float32 (Whisper requires float input)
        waveform = waveform.to(dtype=torch.float32)

        # Ensure sample rate is 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Convert to numpy array for Whisper
        audio_np = waveform.squeeze().numpy()

        # Load Whisper model
        whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")

        # Transcribe audio
        transcription = whisper_model(audio_np)["text"]

        return transcription.strip()

    except Exception as e:
        print(f"Error in audio transcription: {e}")
        return ""


def extract_text_from_image(image_file):
    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'])  # Add other languages if needed

    """
    Extracts text from an uploaded image using EasyOCR.

    Args:
        image_file (InMemoryUploadedFile): Image file uploaded by the user.

    Returns:
        str: Extracted text from the image.
    """
    try:
        # Convert image file to bytes and read with EasyOCR
        image_bytes = io.BytesIO(image_file.read())
        extracted_text = reader.readtext(image_bytes, detail=0)  # detail=0 returns only text, no bounding boxes

        return " ".join(extracted_text).strip()

    except Exception as e:
        print(f"Error in image text extraction: {e}")
        return ""

