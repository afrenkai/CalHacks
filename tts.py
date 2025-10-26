from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa

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
    audio, sampling_rate = librosa.load(audio_path, sr=16000)
    
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    inputs = inputs.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return transcription

if __name__ == "__main__":
    audio_file = input("hello-biden-its-zelensky.mp3")
    
    try:
        print("Transcribing...")
        text = transcribe_audio(audio_file)
        print("\nTranscription:")
        print(text)
    except Exception as e:
        print(f"Error: {e}")
