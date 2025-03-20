import torch
import torchaudio
import soundfile as sf
import os
import numpy as np
import torch.nn.functional as F
from vap.modules.lightning_module import VAPModule

# Load VAP Model
vap_model = VAPModule.load_from_checkpoint(
    "/Users/juliakuhn/Desktop/Helbling_STARTHACK25/src/VAP/example/checkpoints/checkpoint.ckpt"
).model
vap_model.eval()


def isolate_voice_vap(audio_path, output_path="processed_audio.wav"):
    """Load a WAV file, process it with VAP, and save the result."""
    audio, sr = torchaudio.load(audio_path)

    # 🔹 Convert mono to stereo if needed
    if audio.shape[0] == 1:
        audio = torch.cat([audio, audio], dim=0)  # Convert mono to stereo
        print("🔄 Converted mono to stereo!")

    # Add batch dimension
    audio = audio.unsqueeze(0)  # (1, 2, samples)
    print(f"✅ Debug: Corrected audio shape: {audio.shape}")

    # Run through VAP model (predict voice activity)
    with torch.no_grad():
        predictions = vap_model(audio)  # This returns a dictionary

    # Extract `logits`
    if "logits" not in predictions:
        raise ValueError("❌ ERROR: 'logits' key is missing in VAP model output!")

    logits = predictions["logits"]  # Extract actual tensor
    print(f"✅ Extracted logits shape: {logits.shape}")  # Shape: (1, frames, 256)

    # **Compute Speech Mask**
    threshold = 0.3  # Reduce threshold to keep more speech parts
    speech_mask = logits.squeeze(0).mean(dim=-1) > threshold  # Reduce last dim
    print(f"✅ Speech mask shape: {speech_mask.shape}")  # Shape: (frames,)

    # **Resample speech_mask to match audio length**
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
        enhanced_audio = audio[:, :, speech_mask_resampled]

    # Ensure valid shape before saving
    if enhanced_audio.shape[-1] < 1000:
        print("⚠️ WARNING: Output audio is too short! Keeping original audio.")
        enhanced_audio = audio

    # Convert to numpy
    final_audio = enhanced_audio.squeeze(0).cpu().numpy()
    print(f"✅ Final audio shape for saving: {final_audio.shape}")

    # Ensure stereo format (if mono, duplicate channels)
    if final_audio.shape[0] == 1:
        final_audio = np.vstack([final_audio, final_audio])  # Convert mono to stereo
        print("🔄 Converted mono to stereo before saving!")

    # Save the processed audio
    output_file = os.path.join(os.getcwd(), output_path)
    sf.write(output_file, final_audio.T, sr)

    print(f"✅ Processed audio saved at: {output_file}")
    return output_file


# **Test it with an audio file**
if __name__ == "__main__":
    input_audio_path = "/Users/juliakuhn/Desktop/Helbling_STARTHACK25/test_audio.wav"
    output_audio_path = "processed_audio.wav"

    saved_file = isolate_voice_vap(input_audio_path, output_audio_path)
    print(f"🎧 Listen to the processed audio here: {saved_file}")
