import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ssm_cls_head import SSMClassificationHead
from tqdm import tqdm
import json


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels):
        """
        Args:
            texts: List of text strings
            labels: List of integer labels (0=yes, 1=no, 2=maybe)
        """
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch, tokenizer, device):
    texts, labels = zip(*batch)

    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = inputs['input_ids'].to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)

    return input_ids, labels


def train_model(
    train_texts,
    train_labels,
    val_texts=None,
    val_labels=None,
    mamba_model_name="state-spaces/mamba-370m-hf",
    freeze_mamba=True,
    epochs=3,
    batch_size=8,
    learning_rate=2e-4,
    save_path="mamba_classifier.pth"
):
    """
    Train the Mamba-based classifier.

    Args:
        train_texts: List of training text strings
        train_labels: List of training labels (0=yes, 1=no, 2=maybe)
        val_texts: Optional validation texts
        val_labels: Optional validation labels
        mamba_model_name: Pretrained Mamba model to use
        freeze_mamba: Whether to freeze Mamba weights (train only classifier head)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Path to save trained model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {mamba_model_name}")
    model = SSMClassificationHead(
        mamba_model_name=mamba_model_name,
        freeze_mamba=freeze_mamba
    )
    model.to(device)

    train_dataset = TextClassificationDataset(train_texts, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, model.tokenizer, device)
    )

    if val_texts is not None and val_labels is not None:
        val_dataset = TextClassificationDataset(val_texts, val_labels)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, model.tokenizer, device)
        )
    else:
        val_loader = None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for input_ids, labels in progress_bar:
            optimizer.zero_grad()

            logits = model(input_ids)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for input_ids, labels in val_loader:
                    logits = model(input_ids)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total

            print(f"Validation - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")

        print()

    if val_loader is None:
        torch.save(model.state_dict(), save_path)
        print(f"Saved final model to {save_path}")

    return model


def load_data_from_json(json_path):
    """
    Load training data from JSON file.

    Expected format:
    [
        {"text": "Yes, I agree", "label": "yes"},
        {"text": "No way", "label": "no"},
        {"text": "I'm not sure", "label": "maybe"}
    ]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    label_map = {"yes": 0, "no": 1, "maybe": 2, "confusion": 2}

    texts = [item['text'] for item in data]
    labels = [label_map[item['label'].lower()] for item in data]

    return texts, labels


if __name__ == "__main__":
    print("Example training script for Mamba classifier\n")

    train_texts = [
        "Yes, absolutely!",
        "Yes, I agree with that.",
        "Definitely yes.",
        "No, that's incorrect.",
        "No way, I disagree.",
        "Absolutely not.",
        "I'm not sure about that.",
        "Maybe, I need more information.",
        "That's confusing to me.",
        "Hmm, I'm uncertain.",
    ]

    train_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]  # 0=yes, 1=no, 2=maybe

    print("Training with example data...")
    print(f"Train samples: {len(train_texts)}")
    print("\nTo use your own data:")
    print("1. Create a JSON file with format: [{'text': '...', 'label': 'yes/no/maybe'}, ...]")
    print("2. Load with: texts, labels = load_data_from_json('your_data.json')")
    print("3. Call: train_model(texts, labels, save_path='your_model.pth')")
    print("\nFor better results, use at least 100+ examples per class.")
    print("\nStarting training with small example dataset...\n")

    model = train_model(
        train_texts,
        train_labels,
        mamba_model_name="state-spaces/mamba-130m-hf",
        freeze_mamba=True,
        epochs=3,
        batch_size=2,
        learning_rate=1e-3,
        save_path="mamba_classifier_demo.pth"
    )

    print("\nTraining complete! Testing the trained model...")

    test_texts = [
        "Yes, that sounds good",
        "No, I don't think so",
        "I'm confused about this"
    ]

    predictions, probabilities = model.predict(test_texts)

    for text, pred, probs in zip(test_texts, predictions, probabilities):
        print(f"\nText: {text}")
        print(f"Prediction: {pred}")
        print(f"Probabilities: yes={probs[0]:.3f}, no={probs[1]:.3f}, maybe={probs[2]:.3f}")
