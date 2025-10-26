import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from ssm_cls_head import SSMClassificationHead
import os

device = "cuda" if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(os.cpu_count())  # Use all CPU cores
torch.set_grad_enabled(False)  # Disable gradients for inference

# Use wav2vec2-base for much lower memory usage (~360MB vs 8GB for Granite)
model_name = "facebook/wav2vec2-base-960h"

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
model.eval()

ssm_classifier = SSMClassificationHead()

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text using wav2vec2

    Args:
        audio_path: Path to audio file (wav, mp3, flac, etc.)
    Returns:
        Transcribed text as string
    """
    # Load and preprocess audio
    wav, sr = torchaudio.load(audio_path, normalize=True)

    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample to 16kHz if needed (wav2vec2 requires 16kHz)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)

    # Process audio and run inference
    input_values = processor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
    input_values = input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted ids to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription


def classify_transcription(text, model_path=None):
    """
    Classify transcribed text into yes/no/maybe categories

    Args:
        text: Transcribed text string
        model_path: Optional path to trained model weights
    Returns:
        prediction: Predicted class (yes/no/maybe-confusion)
        probabilities: Class probabilities
    """
    if model_path:
        ssm_classifier.load_state_dict(torch.load(model_path, map_location=device))

    predictions, probabilities = ssm_classifier.predict(text, device=device)
    return predictions[0], probabilities[0]


def transcribe_and_classify(audio_path, classifier_model_path=None):
    """
    Transcribe audio and classify the text

    Args:
        audio_path: Path to audio file
        classifier_model_path: Optional path to trained classifier weights
    Returns:
        text: Transcribed text
        prediction: Classification result
        probabilities: Class probabilities
    """
    text = transcribe_audio(audio_path)
    prediction, probabilities = classify_transcription(text, classifier_model_path)
    return text, prediction, probabilities


if __name__ == "__main__":
    audio_file = "do-not-redeem_z7RLKwV.mp3"  
    try:
        print("Transcribing...")
        text = transcribe_audio(audio_file)

        print("\nTranscription:")
        print(text.upper())

        print("\nClassifying transcription...")
        prediction, probs = classify_transcription(text)

        print(f"\nClassification Result: {prediction.upper()}")
        print(f"Probabilities:")
        print(f"  Yes: {probs[0]:.3f}")
        print(f"  No: {probs[1]:.3f}")
        print(f"  Maybe/Confusion: {probs[2]:.3f}")

        print("\n" + "="*60)
        print("Note: Classifier is untrained. For accurate results:")
        print("1. Train the model on labeled data")
        print("2. Save weights and pass path to classifier_model_path")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
