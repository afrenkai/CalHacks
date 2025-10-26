import torch
from torch.utils.data import DataLoader
from diffusion_sampler import DiffusionModel
from diffusion_augmentation import (
    EmbeddingDiffusionNet,
    AugmentedDataLoader,
    train_with_augmentation
)
from ssm_cls_head import SSMClassificationHead
from train_classifier import TextClassificationDataset, collate_fn


def quick_start_example():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = SSMClassificationHead(
        mamba_model_name="state-spaces/mamba-130m-hf",
        freeze_mamba=True
    )
    classifier.to(device)

    embedding_dim = 768
    denoising_net = EmbeddingDiffusionNet(embedding_dim=embedding_dim)
    diffusion = DiffusionModel(denoising_net, timesteps=1000)

    print("Generating 10 samples on-the-fly...")
    with torch.no_grad():
        samples = diffusion.fast_sample(
            shape=(10, embedding_dim),
            device=device,
            num_steps=20
        )

    print(f"Generated samples shape: {samples.shape}")
    print(f"Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")


def full_training_example():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Step 1: Preparing data...")
    train_texts = [
        "Yes, absolutely!",
        "Yes, I agree.",
        "No, that's wrong.",
        "No way.",
        "Maybe, I'm not sure.",
        "I'm confused.",
    ] * 10

    train_labels = [0, 0, 1, 1, 2, 2] * 10  # 0=yes, 1=no, 2=maybe

    dataset = TextClassificationDataset(train_texts, train_labels)

    classifier = SSMClassificationHead(
        mamba_model_name="state-spaces/mamba-130m-hf",
        freeze_mamba=True
    )
    classifier.eval()

    embeddings_list = []
    with torch.no_grad():
        for text in train_texts[:5]:  
            inputs = classifier.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            outputs = classifier.mamba(inputs['input_ids'])
            embedding = outputs.last_hidden_state[:, -1, :]  # Last token, instead of CLS since its custom
            embeddings_list.append(embedding)

    embeddings = torch.cat(embeddings_list, dim=0)
    print(f"Extracted embeddings shape: {embeddings.shape}")

    embedding_dim = embeddings.shape[-1]

    denoising_net = EmbeddingDiffusionNet(embedding_dim=embedding_dim)
    diffusion = DiffusionModel(denoising_net, timesteps=1000)

    diffusion.train()
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

    for epoch in range(3):  
        loss = diffusion(embeddings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/3 - Loss: {loss.item():.4f}")

    diffusion.eval()

    with torch.no_grad():
        synthetic = diffusion.fast_sample(
            shape=(5, embedding_dim),
            device=device,
            num_steps=20
        )

    print(f"Generated {synthetic.shape[0]} synthetic embeddings!")
    print(f"Real data mean: {embeddings.mean():.3f}, std: {embeddings.std():.3f}")
    print(f"Synthetic mean: {synthetic.mean():.3f}, std: {synthetic.std():.3f}")



def continuous_generation_example():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_dim = 768
    denoising_net = EmbeddingDiffusionNet(embedding_dim=embedding_dim)
    diffusion = DiffusionModel(denoising_net, timesteps=1000)
    diffusion.to(device)
    diffusion.eval()


    for batch_idx in range(3):
        print(f"Batch {batch_idx + 1}:")

        with torch.no_grad():
            samples = diffusion.fast_sample(
                shape=(8, embedding_dim),
                device=device,
                num_steps=10
            )

        print(f"  Generated {samples.shape[0]} samples in real-time")
        print(f"  Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")



