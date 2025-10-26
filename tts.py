import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from ssm_cls_head import SSMClassificationHead
import os

device = "cuda" if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(os.cpu_count())  # Use all CPU cores
torch.set_grad_enabled(False)  # Disable gradients for inference

model_name = "ibm-granite/granite-speech-3.3-8b"

processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name,
    device_map=device,
    low_cpu_mem_usage=True,  # Reduce memory usage
)


ssm_classifier = SSMClassificationHead()

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text
    
    Args:
        audio_path: Path to audio file (wav, mp3, flac, etc.)
    Returns:
        Transcribed text as string
    """
    wav, sr = torchaudio.load(audio_path, normalize=True)
    
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    
    system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
    user_prompt = "<|audio|>can you transcribe the speech into a written format?"
    chat = [
        dict(role="system", content=system_prompt),
        dict(role="user", content=user_prompt),
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    model_inputs = processor(prompt, wav, device=device, return_tensors="pt")
    model_outputs = model.generate(
        **model_inputs,
        max_new_tokens=200,
        do_sample=False,
        use_cache=True,  # Enable KV caching
        pad_token_id=tokenizer.pad_token_id
    )
    
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
    output_text = tokenizer.batch_decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True
    )
    
    return output_text[0]


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
    audio_file = "hello-biden-its-zelensky.mp3"

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
