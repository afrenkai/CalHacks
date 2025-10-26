import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ibm-granite/granite-speech-3.3-8b"

processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name, device_map=device, torch_dtype=torch.bfloat16
)

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text
    
    Args:
        audio_path: Path to audio file (wav, mp3, flac, etc.)
    Returns:
        Transcribed text as string
    """
    # Load audio - ensure it's mono and 16kHz
    wav, sr = torchaudio.load(audio_path, normalize=True)
    
    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
    
    # Create text prompt
    system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
    user_prompt = "<|audio|>can you transcribe the speech into a written format?"
    chat = [
        dict(role="system", content=system_prompt),
        dict(role="user", content=user_prompt),
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    # Run the processor+model
    model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
    model_outputs = model.generate(**model_inputs, max_new_tokens=200, do_sample=False, num_beams=1)
    
    # Transformers includes the input IDs in the response.
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
    output_text = tokenizer.batch_decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True
    )
    
    return output_text[0]

if __name__ == "__main__":
    audio_file = "hello-biden-its-zelensky.mp3"
    
    try:
        print("Transcribing...")
        text = transcribe_audio(audio_file)
        
        print("\nTranscription:")
        print(text.upper())
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
