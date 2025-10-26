import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, MambaModel, AutoConfig


class SSMClassificationHead(nn.Module):
    """
    Mamba-based text classifier for yes/no/maybe responses.
    Uses pretrained Mamba model from state-spaces.
    """
    def __init__(self,
                 mamba_model_name="state-spaces/mamba-370m-hf",
                 dropout=0.1,
                 num_classes=3,
                 freeze_mamba=False):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(mamba_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.mamba = MambaModel.from_pretrained(mamba_model_name)

        if freeze_mamba:
            for param in self.mamba.parameters():
                param.requires_grad = False

        d_model = self.mamba.config.hidden_size

        self.layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.num_classes = num_classes
        self.class_names = ["yes", "no", "maybe/confusion"]

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len) tensor of token ids
        Returns:
            logits: (batch_size, num_classes)
        """
        mamba_outputs = self.mamba(input_ids)
        hidden_states = mamba_outputs.last_hidden_state  # (batch, seq_len, d_model)

        pooled = hidden_states[:, -1, :]  # (batch, d_model)

        pooled = self.layer_norm(pooled)
        logits = self.classifier(pooled)

        return logits

    def predict(self, texts, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Predict class for input texts.

        Args:
            texts: str or List[str]
            device: device to run on
        Returns:
            predictions: List of predicted class names
            probabilities: List of probability distributions
        """
        self.eval()
        self.to(device)

        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(device)

        with torch.no_grad():
            logits = self.forward(input_ids)
            probs = F.softmax(logits, dim=-1)
            pred_classes = torch.argmax(probs, dim=-1)

        predictions = [self.class_names[idx.item()] for idx in pred_classes]
        probabilities = probs.cpu().numpy()

        return predictions, probabilities


def classify_text(text, model_path=None, mamba_model_name="state-spaces/mamba-370m-hf"):
    """
    Convenience function to classify text.

    Args:
        text: Input text string or list of strings
        model_path: Path to saved model weights (optional)
        mamba_model_name: Pretrained Mamba model to use
    Returns:
        predictions: List of predicted classes
        probabilities: Numpy array of class probabilities
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SSMClassificationHead(mamba_model_name=mamba_model_name)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))

    predictions, probabilities = model.predict(text, device=device)

    return predictions, probabilities


if __name__ == "__main__":
    print("Initializing Mamba-based Classification Head...")
    print("Loading pretrained Mamba model (this may take a moment)...")

    # Create model - you can use different sizes:
    # "state-spaces/mamba-130m-hf" (smallest, fastest)
    
    model = SSMClassificationHead(mamba_model_name="state-spaces/mamba-370m-hf")
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Example texts
    test_texts = [
        "Yes, I completely agree with that.",
        "No, that's not correct.",
        "I'm not sure, maybe we should check again.",
        "Absolutely!",
        "I don't think so.",
        "Hmm, I'm confused about that.",
    ]

    print("\nTesting classification (pretrained Mamba + untrained head):")
    predictions, probabilities = model.predict(test_texts)

    for text, pred, probs in zip(test_texts, predictions, probabilities):
        print(f"\nText: {text}")
        print(f"Prediction: {pred}")
        print(f"Probabilities: yes={probs[0]:.3f}, no={probs[1]:.3f}, maybe={probs[2]:.3f}")

