import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from diffusion_sampler import DiffusionModel, SimpleDenoisingUNet
from ssm_cls_head import SSMClassificationHead
import numpy as np
from tqdm import tqdm


class EmbeddingDiffusionNet(nn.Module):
    def __init__(self, embedding_dim: int = 768, time_emb_dim: int = 256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(embedding_dim + time_emb_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x, t):
        # x: (batch, embedding_dim)
        # t: (batch,) timestep indices

        t_normalized = t.float() / 1000.0
        t_emb = self.time_mlp(t_normalized.unsqueeze(-1))  # (batch, time_emb_dim)

        h = torch.cat([x, t_emb], dim=-1)

        return self.net(h)


class AugmentedDataLoader:
    def __init__(self,
                 base_dataloader: DataLoader,
                 diffusion_model: DiffusionModel,
                 augmentation_ratio: float = 0.5,
                 device: str = "cuda"):
        """
        Args:
            base_dataloader: Original data loader
            diffusion_model: Trained diffusion model for generating samples
            augmentation_ratio: Ratio of synthetic to real samples (0.5 = 50% synthetic)
            device: Device to generate samples on
        """
        self.base_dataloader = base_dataloader
        self.diffusion_model = diffusion_model
        self.augmentation_ratio = augmentation_ratio
        self.device = device

    def __iter__(self):
        for batch in self.base_dataloader:
            if isinstance(batch, (list, tuple)):
                real_data, labels = batch
            else:
                real_data = batch
                labels = None

            real_data = real_data.to(self.device)
            batch_size = real_data.shape[0]

            num_synthetic = int(batch_size * self.augmentation_ratio)

            if num_synthetic > 0:
                with torch.no_grad():
                    synthetic_data = self.diffusion_model.fast_sample(
                        shape=(num_synthetic,) + real_data.shape[1:],
                        device=self.device,
                        num_steps=20  # Fast sampling
                    )

                combined_data = torch.cat([real_data, synthetic_data], dim=0)

                if labels is not None:
                    synthetic_labels = torch.randint_like(
                        labels[:num_synthetic],
                        low=0,
                        high=labels.max().item() + 1
                    )
                    combined_labels = torch.cat([labels, synthetic_labels], dim=0)
                    yield combined_data, combined_labels
                else:
                    yield combined_data
            else:
                if labels is not None:
                    yield real_data, labels
                else:
                    yield real_data

    def __len__(self):
        return len(self.base_dataloader)


class ConditionalDiffusionNet(nn.Module):
    def __init__(self, embedding_dim: int = 768, num_classes: int = 3, time_emb_dim: int = 256):
        super().__init__()

        self.num_classes = num_classes

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        self.net = nn.Sequential(
            nn.Linear(embedding_dim + time_emb_dim * 2, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x, t, class_labels):
        t_normalized = t.float() / 1000.0
        t_emb = self.time_mlp(t_normalized.unsqueeze(-1))

        c_emb = self.class_emb(class_labels)

        h = torch.cat([x, t_emb, c_emb], dim=-1)

        return self.net(h)


def generate_class_balanced_samples(diffusion_model: DiffusionModel,
                                   class_label: int,
                                   num_samples: int,
                                   embedding_dim: int,
                                   device: str = "cuda"):
    if hasattr(diffusion_model.model, 'num_classes'):
        with torch.no_grad():
            noise = torch.randn(num_samples, embedding_dim, device=device)
            class_labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
            x = noise
            for t in reversed(range(diffusion_model.timesteps)):
                t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
                predicted_noise = diffusion_model.model(x, t_tensor, class_labels)
                betas_t = diffusion_model.scheduler.betas[t].to(device)
                sqrt_one_minus_alphas_cumprod_t = diffusion_model.scheduler.sqrt_one_minus_alphas_cumprod[t].to(device)
                sqrt_recip_alphas_t = diffusion_model.scheduler.sqrt_recip_alphas[t].to(device)

                model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

                if t > 0:
                    noise = torch.randn_like(x)
                    posterior_variance_t = diffusion_model.scheduler.posterior_variance[t].to(device)
                    x = model_mean + torch.sqrt(posterior_variance_t) * noise
                else:
                    x = model_mean

            return x
    else:
        return diffusion_model.fast_sample(
            shape=(num_samples, embedding_dim),
            device=device,
            num_steps=50
        )


def train_with_augmentation(classifier: SSMClassificationHead,
                           train_loader: DataLoader,
                           diffusion_model: DiffusionModel,
                           epochs: int = 10,
                           augmentation_ratio: float = 0.5,
                           lr: float = 1e-4,
                           device: str = "cuda"):
    classifier.to(device)
    diffusion_model.to(device)
    diffusion_model.eval()  # Frozen for generation, assumes requires_grad = False and not detached

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    aug_loader = AugmentedDataLoader(
        train_loader,
        diffusion_model,
        augmentation_ratio=augmentation_ratio,
        device=device
    )

    for epoch in range(epochs):
        classifier.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(aug_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for input_ids, labels in progress_bar:
            optimizer.zero_grad()

            logits = classifier(input_ids)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        avg_loss = epoch_loss / len(train_loader)
        acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")

    return classifier


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    embedding_dim = 768
    denoising_net = EmbeddingDiffusionNet(embedding_dim=embedding_dim)
    diffusion = DiffusionModel(denoising_net, timesteps=1000, schedule_type="cosine")
    diffusion.to(device)

    print(f"Model parameters: {sum(p.numel() for p in diffusion.parameters()):,}")

    with torch.no_grad():
        samples = diffusion.fast_sample(
            shape=(8, embedding_dim),
            device=device,
            num_steps=20
        )

    print(f"Generated {samples.shape[0]} embedding vectors")
    print(f"Shape: {samples.shape}")
    print(f"Mean: {samples.mean():.3f}, Std: {samples.std():.3f}\n")
    conditional_net = ConditionalDiffusionNet(
        embedding_dim=embedding_dim,
        num_classes=3  # yes, no, maybe
    )

    print(f"Conditional model parameters: {sum(p.numel() for p in conditional_net.parameters()):,}")

