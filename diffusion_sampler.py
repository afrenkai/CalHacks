import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm


class DiffusionScheduler:
    """
    Variance schedule for diffusion process.
    Implements linear and cosine schedules.
    """
    def __init__(self,
                 timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 schedule_type: str = "linear"):
        self.timesteps = timesteps

        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


class DenoisingUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 dim: int = 64,
                 time_emb_dim: int = 256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.enc1 = self._conv_block(in_channels, dim, time_emb_dim)
        self.enc2 = self._conv_block(dim, dim * 2, time_emb_dim)
        self.enc3 = self._conv_block(dim * 2, dim * 4, time_emb_dim)

        self.bottleneck = self._conv_block(dim * 4, dim * 4, time_emb_dim)

        self.dec3 = self._conv_block(dim * 8, dim * 2, time_emb_dim)
        self.dec2 = self._conv_block(dim * 4, dim, time_emb_dim)
        self.dec1 = self._conv_block(dim * 2, dim, time_emb_dim)

        self.final = nn.Conv1d(dim, out_channels, 1)

    def _conv_block(self, in_ch, out_ch, time_emb_dim):
        return nn.ModuleDict({
            'conv1': nn.Conv1d(in_ch, out_ch, 3, padding=1),
            'conv2': nn.Conv1d(out_ch, out_ch, 3, padding=1),
            'norm1': nn.GroupNorm(8, out_ch),
            'norm2': nn.GroupNorm(8, out_ch),
            'time_emb': nn.Linear(time_emb_dim, out_ch),
        })

    def _apply_block(self, x, t_emb, block):
        h = block['conv1'](x)
        h = block['norm1'](h)
        h = F.silu(h)

        # Add time embedding
        t_emb_proj = block['time_emb'](t_emb)[:, :, None]
        h = h + t_emb_proj

        h = block['conv2'](h)
        h = block['norm2'](h)
        h = F.silu(h)
        return h

    def forward(self, x, t):
        t_normalized = t.float() / 1000.0
        t_emb = self.time_mlp(t_normalized.unsqueeze(-1))
        e1 = self._apply_block(x, t_emb, self.enc1)
        e2 = self._apply_block(F.max_pool1d(e1, 2), t_emb, self.enc2)
        e3 = self._apply_block(F.max_pool1d(e2, 2), t_emb, self.enc3)
        b = self._apply_block(F.max_pool1d(e3, 2), t_emb, self.bottleneck)
        d3 = self._apply_block(
            torch.cat([F.interpolate(b, size=e3.shape[-1]), e3], dim=1),
            t_emb, self.dec3
        )
        d2 = self._apply_block(
            torch.cat([F.interpolate(d3, size=e2.shape[-1]), e2], dim=1),
            t_emb, self.dec2
        )
        d1 = self._apply_block(
            torch.cat([F.interpolate(d2, size=e1.shape[-1]), e1], dim=1),
            t_emb, self.dec1
        )

        return self.final(d1)


class DiffusionModel(nn.Module):
    def __init__(self,
                 denoising_model: nn.Module,
                 timesteps: int = 1000,
                 schedule_type: str = "linear"):
        super().__init__()

        self.model = denoising_model
        self.scheduler = DiffusionScheduler(timesteps, schedule_type=schedule_type)
        self.timesteps = timesteps

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Forward diffusion: add noise to data.
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.scheduler.sqrt_alphas_cumprod[t].to(x_0.device)
        sqrt_one_minus_alphas_cumprod_t = self.scheduler.sqrt_one_minus_alphas_cumprod[t].to(x_0.device)
        while len(sqrt_alphas_cumprod_t.shape) < len(x_0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def p_losses(self, x_0: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x_0)
        x_noisy, _ = self.q_sample(x_0, t, noise=noise)

        predicted_noise = self.model(x_noisy, t)

        loss = F.mse_loss(predicted_noise, noise)
        #derived from the 
        return loss

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, t_tensor: torch.Tensor):
        """
        Single reverse diffusion step.
        Sample from p(z_{t-1} | z_t).
        """
        betas_t = self.scheduler.betas[t].to(x.device)
        sqrt_one_minus_alphas_cumprod_t = self.scheduler.sqrt_one_minus_alphas_cumprod[t].to(x.device)
        sqrt_recip_alphas_t = self.scheduler.sqrt_recip_alphas[t].to(x.device)

        while len(sqrt_recip_alphas_t.shape) < len(x.shape):
            betas_t = betas_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
            sqrt_recip_alphas_t = sqrt_recip_alphas_t.unsqueeze(-1)

        predicted_noise = self.model(x, t_tensor)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self.scheduler.posterior_variance[t].to(x.device)
            while len(posterior_variance_t.shape) < len(x.shape):
                posterior_variance_t = posterior_variance_t.unsqueeze(-1)

            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], device: str = "cuda", show_progress: bool = True):
        x = torch.randn(shape, device=device)
        timesteps_iter = reversed(range(self.timesteps))
        if show_progress:
            timesteps_iter = tqdm(list(timesteps_iter), desc="Sampling")

        for t in timesteps_iter:
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t, t_tensor)

        return x

    @torch.no_grad()
    def fast_sample(self, shape: Tuple[int, ...], device: str = "cuda", num_steps: int = 50):
        #DDIM, so faster since no need for E_theta
        step_size = self.timesteps // num_steps
        timesteps = list(range(0, self.timesteps, step_size))[::-1]

        x = torch.randn(shape, device=device)

        for t in tqdm(timesteps, desc="Fast sampling"):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t, t_tensor)

        return x

    def forward(self, x_0: torch.Tensor):
        batch_size = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        loss = self.p_losses(x_0, t)
        return loss


def train_diffusion(model: DiffusionModel,
                   dataloader,
                   epochs: int = 100,
                   lr: float = 1e-4,
                   device: str = "cuda",
                   save_path: str = "diffusion_model.pth",
                   sample_every: int = 10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(device)

            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

        # Generate samples on the fly
        if (epoch + 1) % sample_every == 0:
            print("Generating samples...")
            model.eval()
            samples = model.fast_sample(
                shape=(4, batch.shape[1], batch.shape[2]),
                device=device,
                num_steps=50
            )
            print(f"Generated {samples.shape[0]} samples with shape {samples.shape}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    denoising_net = DenoisingUNet(
        in_channels=1,
        out_channels=1,
        dim=64,
        time_emb_dim=256
    )

    diffusion = DiffusionModel(
        denoising_model=denoising_net,
        timesteps=1000,
        schedule_type="cosine"
    )

    diffusion.to(device)

    print(f"Model parameters: {sum(p.numel() for p in diffusion.parameters()):,}")
    with torch.no_grad():
        samples = diffusion.fast_sample(
            shape=(4, 1, 256),  # (batch, channels, length)
            device=device,
            num_steps=50
        )

    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample statistics - Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")
