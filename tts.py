# from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
# import torch
# import librosa
# processor = AutoProcessor.from_pretrained("ibm-granite/granite-speech-3.3-8b")
# model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b")
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)
#
# def transcribe_audio(audio_path):
#     """
#     Transcribe audio file to text
#
#     Args:
#         audio_path: Path to audio file (wav, mp3, flac, etc.)
#     Returns:
#         Transcribed text as string
#     """
#     # Load audio at 16kHz
#     audio, sampling_rate = librosa.load(audio_path, sr=16000)
#
#     # Process audio - pass only the audio array
#     inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
#
#     # Move inputs to device
#     input_features = inputs.input_features.to(device)
#
#     # Generate transcription
#     with torch.no_grad():
#         generated_ids = model.generate(input_features)
#
#     # Decode the generated ids
#     transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#
#     return transcription
#
# if __name__ == "__main__":
#     audio_file = "hello-biden-its-zelensky.mp3"
#
#     try:
#         print("Transcribing...")
#         text = transcribe_audio(audio_file)
#
#         print("\nTranscription:")
#         print(text)
#     except Exception as e:
#         print(f"Error: {e}")
#
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("ibm-granite/granite-speech-3.3-8b")
model = AutoModelForSpeechSeq2Seq.from_pretrained("ibm-granite/granite-speech-3.3-8b")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text
    
    Args:
        audio_path: Path to audio file (wav, mp3, flac, etc.)
    Returns:
        Transcribed text as string
    """
    # Load audio at 16kHz
    audio, sampling_rate = librosa.load(audio_path, sr=16000)
    
    # Process audio - use 'audio' parameter name
    inputs = processor(audio=audio, sampling_rate=sampling_rate, return_tensors="pt")
    
    # Move all inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    
    # Decode the generated ids
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return transcription

if __name__ == "__main__":
    audio_file = "hello-biden-its-zelensky.mp3"
    
    try:
        print("Transcribing...")
        text = transcribe_audio(audio_file)
        
        print("\nTranscription:")
        print(text)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
