import uuid
import json
import os
import io
import torch
import torchaudio
import soundfile as sf
import numpy as np
import torch.nn.functional as F

from flask import Flask, request, jsonify, redirect 
import azure.cognitiveservices.speech as speechsdk
from flask_sock import Sock
from flask_cors import CORS
from flasgger import Swagger

from openai import OpenAI 
import openai
from vap.modules.lightning_module import VAPModule

openai.api_key = os.getenv("OPENAI_API_KEY")

AZURE_SPEECH_KEY = "See https://starthack.eu/#/case-details?id=21, Case Description"
AZURE_SPEECH_REGION = "switzerlandnorth"
OPENAI_KEY = "See https://starthack.eu/#/case-details?id=21, Case Description"
client = OpenAI(api_key=OPENAI_KEY)

app = Flask(__name__)
sock = Sock(app)
cors = CORS(app)
swagger = Swagger(app)

sessions = {}

# Load VAP model once at startup
print("Loading VAP Model...")
try:
    vap_model = VAPModule.load_from_checkpoint(
        "/Users/juliakuhn/Desktop/Helbling_STARTHACK25/src/VAP/example/checkpoints/checkpoint.ckpt"
    ).model  # Ensure we access the actual model inside VAPModule

    vap_model.eval()
    print("✅ VAP Model Loaded Successfully.")
except Exception as e:
    print(f"🚨 Error loading VAP Model: {e}")
    exit(1)  # Stop execution if the model fails to load


def isolate_voice_vap(audio_bytes):
    """
    Use VAP to extract the main speaker's voice and remove background noise.

    Parameters:
    - audio_bytes (bytes): The raw audio data in bytes.

    Returns:
    - bytes: Processed audio with main speaker isolated.
    """
    print("🔄 Isolating voice with VAP...")
    
    # Load audio from bytes
    audio_buffer = io.BytesIO(audio_bytes)
    audio, sr = sf.read(audio_buffer, dtype="float32")

    # Ensure audio is stereo (VAP expects 2 channels)
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio], axis=0)  # Convert mono to stereo
        print("🔄 Converted mono to stereo!")

    # Convert to PyTorch tensor & add batch dimension
    audio_tensor = torch.tensor(audio).unsqueeze(0)  # Shape: (1, 2, samples)
    print(f"✅ Debug: Input audio shape for VAP: {audio_tensor.shape}")

    # Run through VAP model to predict voice activity
    with torch.no_grad():
        predictions = vap_model(audio_tensor)  # Returns dictionary

    # Extract `logits`
    if "logits" not in predictions:
        raise ValueError("❌ ERROR: 'logits' key is missing in VAP model output!")

    logits = predictions["logits"]  # Extract actual tensor
    print(f"✅ Extracted logits shape: {logits.shape}")  # Shape: (1, frames, 256)

    # Compute Speech Mask
    threshold = 0.3  # Reduce threshold to keep more speech parts
    speech_mask = logits.squeeze(0).mean(dim=-1) > threshold  # Reduce last dim
    print(f"✅ Speech mask shape: {speech_mask.shape}")  # Shape: (frames,)

    # Resample speech_mask to match audio length
    speech_mask_resampled = F.interpolate(
        speech_mask.unsqueeze(0).unsqueeze(0).float(),
        size=audio.shape[-1],  # Match audio length
        mode="linear"
    ).squeeze() > 0.2  # Keep some low-confidence speech
    print(f"✅ Resampled mask shape: {speech_mask_resampled.shape}")

    # Ensure at least some speech is kept
    if speech_mask_resampled.sum() == 0:
        print("⚠️ WARNING: Speech mask removed everything! Keeping original audio.")
        enhanced_audio = audio
    else:
        enhanced_audio = audio[:, speech_mask_resampled]

    # Ensure valid shape before saving
    if enhanced_audio.shape[-1] < 1000:
        print("⚠️ WARNING: Output audio is too short! Keeping original audio.")
        enhanced_audio = audio

    # Convert to bytes
    output_buffer = io.BytesIO()
    sf.write(output_buffer, enhanced_audio.T, sr, format="wav")
    
    print(f"✅ Processed audio ready for API response")
    return output_buffer.getvalue()


@app.route('/')
def home():
    return redirect('/apidocs') 


@app.route("/chats/<chat_session_id>/sessions/<session_id>/wav", methods=["POST"])
def upload_audio_chunk(chat_session_id, session_id):
    """
    Upload an audio chunk (expected 16kb, ~0.5s of WAV data).
    The chunk is processed with VAP for voice isolation.

    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: ID of the voice input session
      - name: audio_chunk
        in: body
        required: true
        schema:
          type: string
          format: binary
          description: Raw WAV audio data
    responses:
      200:
        description: Audio chunk processed successfully
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    audio_data = request.get_data()
    print("🎙️ Audio data received")

    # Apply VAP-based voice isolation
    clean_audio = isolate_voice_vap(audio_data)
    print("🎧 Audio isolated")

    # Store processed audio in session buffer
    if sessions[session_id]["audio_buffer"] is not None:
        sessions[session_id]["audio_buffer"] += clean_audio
    else:
        sessions[session_id]["audio_buffer"] = clean_audio

    return jsonify({"status": "audio_chunk_received"})


@app.route("/chats/<chat_session_id>/sessions/<session_id>", methods=["DELETE"])
def close_session(chat_session_id, session_id):
    """
    Close the session (stop recognition, process audio, and transcribe speech).

    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: The ID of the session to close
    responses:
      200:
        description: Session successfully closed
    """
    print("Closing session...")
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    if sessions[session_id]["audio_buffer"] is not None:
        # Apply VAP before Whisper transcription
        clean_audio = isolate_voice_vap(sessions[session_id]["audio_buffer"])

    # Remove session
    sessions.pop(session_id, None)
    return jsonify({"status": "session_closed"})


if __name__ == "__main__":
    print("Starting Flask API...")
    app.run(debug=True, host="0.0.0.0", port=5000)
