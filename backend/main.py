from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import sys
import os
from pathlib import Path
import tempfile

# Add parent directory to path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))
from ssm_cls_head import SSMClassificationHead

app = FastAPI(title="Audio Transcription & Classification API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class TranscriptionResponse(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]

class TranscriptionAndClassificationResponse(BaseModel):
    text: str
    prediction: str
    probabilities: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    message: str

# Global model initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count())
torch.set_grad_enabled(False)

model_name = "ibm-granite/granite-speech-3.3-8b"
processor = None
tokenizer = None
model = None
ssm_classifier = None

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global processor, tokenizer, model, ssm_classifier

    print(f"Loading models on device: {device}")
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    ssm_classifier = SSMClassificationHead()
    print("Models loaded successfully")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Audio Transcription & Classification API is running"
    )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_endpoint(audio: UploadFile = File(...)):
    """
    Transcribe audio file to text

    Args:
        audio: Audio file (wav, mp3, flac, etc.)

    Returns:
        Transcription result
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load and preprocess audio
        wav, sr = torchaudio.load(tmp_path, normalize=True)

        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)

        # Prepare prompt
        system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        chat = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=user_prompt),
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # Process and generate
        model_inputs = processor(prompt, wav, device=device, return_tensors="pt")
        model_outputs = model.generate(
            **model_inputs,
            max_new_tokens=200,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode output
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
        output_text = tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )

        # Clean up temp file
        os.unlink(tmp_path)

        return TranscriptionResponse(text=output_text[0])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text_endpoint(text: str):
    """
    Classify text into yes/no/maybe categories

    Args:
        text: Text to classify

    Returns:
        Classification result with probabilities
    """
    try:
        predictions, probabilities = ssm_classifier.predict(text, device=device)

        return ClassificationResponse(
            prediction=predictions[0],
            probabilities={
                "yes": float(probabilities[0][0]),
                "no": float(probabilities[0][1]),
                "maybe_confusion": float(probabilities[0][2])
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe-and-classify", response_model=TranscriptionAndClassificationResponse)
async def transcribe_and_classify_endpoint(audio: UploadFile = File(...)):
    """
    Transcribe audio and classify the text in one request

    Args:
        audio: Audio file (wav, mp3, flac, etc.)

    Returns:
        Transcription and classification results
    """
    try:
        # First transcribe
        transcription_result = await transcribe_audio_endpoint(audio)
        text = transcription_result.text

        # Then classify
        predictions, probabilities = ssm_classifier.predict(text, device=device)

        return TranscriptionAndClassificationResponse(
            text=text,
            prediction=predictions[0],
            probabilities={
                "yes": float(probabilities[0][0]),
                "no": float(probabilities[0][1]),
                "maybe_confusion": float(probabilities[0][2])
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
