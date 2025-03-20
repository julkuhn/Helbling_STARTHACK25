import uuid
import json
import os
import io
import torch
import torchaudio
import soundfile as sf
import numpy as np 


from flask import Flask, request, jsonify, redirect 
import azure.cognitiveservices.speech as speechsdk
from flask_sock import Sock
from flask_cors import CORS
from flasgger import Swagger

from openai import OpenAI 
import io
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
vap_model = VAPModule.load_from_checkpoint("/Users/juliakuhn/Desktop/Helbling_STARTHACK25/src/VAP/example/checkpoints/checkpoint.ckpt")
vap_model.eval()
print("VAP Model Loaded Successfully.")


def isolate_voice_vap(audio_bytes):
    """
    Use VAP to extract the main speaker's voice and remove background noise.

    Parameters:
    - audio_bytes (bytes): The raw audio data in bytes.

    Returns:
    - bytes: Processed audio with main speaker isolated.
    """
    audio_buffer = io.BytesIO(audio_bytes)
    audio, sr = sf.read(audio_buffer, dtype="float32")

    # Ensure audio is stereo (VAP expects 2 channels)
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio], axis=0)  # Convert mono to stereo

    # Convert to PyTorch tensor & add batch dimension
    audio_tensor = torch.tensor(audio).unsqueeze(0)  # Shape: (1, 2, samples)

    print(f"Debug: Input audio shape for VAP: {audio_tensor.shape}")  # Should be (1, 2, N)

    # Run through VAP model to predict voice activity
    with torch.no_grad():
        predictions = vap_model(audio_tensor)  # VAP expects stereo input

    # Apply thresholding to keep only main voice regions
    threshold = 0.6  
    speech_mask = predictions.squeeze(0) > threshold
    enhanced_audio = audio[:, speech_mask]  # Apply mask

    # Convert back to bytes
    output_buffer = io.BytesIO()
    sf.write(output_buffer, enhanced_audio.T, sr, format="wav")

    return output_buffer.getvalue()


@app.route('/')
def home():
    return redirect('/apidocs') 

def transcribe_whisper(audio_recording):
    """
    Transcribes speech from audio using OpenAI Whisper.

    Parameters:
    - audio_recording (bytes): Processed audio data.

    Returns:
    - str: Transcription of the audio.
    """
    audio_file = io.BytesIO(audio_recording)
    audio_file.name = "audio.wav"

    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    
    print(f"Transcription: {transcription.text}")
    return transcription.text


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

    # Apply VAP-based voice isolation
    clean_audio = isolate_voice_vap(audio_data)

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
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    if sessions[session_id]["audio_buffer"] is not None:
        # Apply VAP before Whisper transcription
        clean_audio = isolate_voice_vap(sessions[session_id]["audio_buffer"])

        # Transcribe with Whisper
        text = transcribe_whisper(clean_audio)

        ws = sessions[session_id].get("websocket")
        if ws:
            message = {"event": "recognized", "text": text, "language": sessions[session_id]["language"]}
            ws.send(json.dumps(message))

    # Remove session
    sessions.pop(session_id, None)
    return jsonify({"status": "session_closed"})


@app.route('/chats/<chat_session_id>/set-memories', methods=['POST'])
def set_memories(chat_session_id):
    """
    Stores memories for a chat session.

    ---
    tags:
      - Memories
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier of the chat session.
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            chat_history:
              type: array
              items:
                type: object
                properties:
                  text:
                    type: string
                    description: The chat message text.
    responses:
      200:
        description: Memory set successfully.
    """
    chat_history = request.get_json()

    if not isinstance(chat_history, list) or len(chat_history) == 0:
        return jsonify({"error": "Invalid chat history format"}), 400

    print(f"Memory stored: {chat_history[-1]['text']}")
    return jsonify({"success": "1"})


@app.route('/chats/<chat_session_id>/get-memories', methods=['GET'])
def get_memories(chat_session_id):
    """
    Retrieves stored memories.

    ---
    tags:
      - Memories
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier of the chat session.
    responses:
      200:
        description: Successfully retrieved memories for the chat session.
    """
    print(f"Retrieving memories for session {chat_session_id}")
    return jsonify({"memories": "The guest typically orders menu 1 and a glass of sparkling water."})


if __name__ == "__main__":
    print("Starting Flask API...")
    app.run(debug=True, host="0.0.0.0", port=5000)
