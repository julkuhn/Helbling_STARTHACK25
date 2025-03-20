import torch
import torchaudio
import soundfile as sf
import io
from vap.modules.lightning_module import VAPModule

# Load the VAP model
vap_model = VAPModule.load_from_checkpoint("/Users/juliakuhn/Desktop/Helbling_STARTHACK25/src/VAP/example/checkpoints/checkpoint.ckpt")
vap_model.eval()  # Set model to evaluation mode

def isolate_voice_vap(audio_path):
    """Load a WAV file, process it with VAP, and save the result."""
    audio, sr = torchaudio.load(audio_path)

    # 🔹 If audio is mono, duplicate it to create stereo
    if audio.shape[0] == 1:
        audio = torch.cat([audio, audio], dim=0)  # Convert mono to stereo
        print("🔄 Converted mono to stereo!")

    # Add batch dimension for model input
    audio = audio.unsqueeze(0)  # (1, 2, samples)
    print(f"✅ Debug: Corrected audio shape: {audio.shape}")  # Debugging line

    # Run through VAP model (predict voice activity)
    with torch.no_grad():
        predictions = vap_model(audio)  # ✅ Now correctly passing stereo audio

    # Apply thresholding to filter non-speech regions
    threshold = 0.6
    speech_mask = predictions.squeeze(0) > threshold
    enhanced_audio = audio[:, :, speech_mask]

    # Save processed audio
    output_path = "processed_audio.wav"
    sf.write(output_path, enhanced_audio.squeeze(0).numpy(), sr)
    print(f"✅ Processed audio saved at: {output_path}")

