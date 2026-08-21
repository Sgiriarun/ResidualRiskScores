"""Small neural representation learners for the average-day AI benchmark."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import MINUTES, set_seed, standardize, standardizer_fit


class ActivityAutoencoder(nn.Module):
    def __init__(self, dimension: int, hidden: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(MINUTES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dimension),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.ReLU(),
            nn.Linear(hidden, MINUTES),
        )

    def forward(self, values: torch.Tensor):
        embedding = self.encoder(values)
        return self.decoder(embedding), embedding


class ContrastiveActivityEncoder(nn.Module):
    def __init__(self, dimension: int, hidden: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(MINUTES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dimension),
        )
        self.projector = nn.Sequential(
            nn.Linear(dimension, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )


def block_mask(
    batch_size: int,
    blocks: int,
    block_minutes: int,
    generator: torch.Generator,
) -> torch.Tensor:
    mask = torch.zeros((batch_size, MINUTES), dtype=torch.bool)
    starts = torch.randint(0, MINUTES, (batch_size, blocks), generator=generator)
    offsets = torch.arange(block_minutes)[None, None, :]
    locations = (starts[:, :, None] + offsets) % MINUTES
    rows = torch.arange(batch_size)[:, None, None].expand_as(locations)
    mask[rows.reshape(-1), locations.reshape(-1)] = True
    return mask


def corrupt_batch(
    values: torch.Tensor,
    config: dict,
    generator: torch.Generator,
    contrastive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    generated_mask = block_mask(
        len(values),
        int(config["mask_blocks"]),
        int(config["mask_block_minutes"]),
        generator,
    )
    use_mask = bool(config.get("contrastive_use_mask", True)) if contrastive else True
    mask = generated_mask if use_mask else torch.zeros_like(values, dtype=torch.bool)
    output = values.clone()
    output[mask] = 0.0
    if contrastive:
        scale = 0.9 + 0.2 * torch.rand((len(values), 1), generator=generator)
        noise = 0.03 * torch.randn(output.shape, generator=generator)
        shift = int(torch.randint(-30, 31, (1,), generator=generator).item())
        if bool(config.get("contrastive_use_scale", True)):
            output = output * scale
        if bool(config.get("contrastive_use_noise", True)):
            output = output + noise
        if bool(config.get("contrastive_use_shift", True)):
            output = torch.roll(output, shifts=shift, dims=1)
    return output, mask


def standardized_profiles(train: np.ndarray, test: np.ndarray):
    mean, scale = standardizer_fit(train)
    return (
        standardize(train, mean, scale).astype(np.float32),
        standardize(test, mean, scale).astype(np.float32),
    )


def encode(model: nn.Module, train: np.ndarray, test: np.ndarray):
    model.eval()
    with torch.no_grad():
        train_embedding = model.encoder(torch.as_tensor(train)).numpy()
        test_embedding = model.encoder(torch.as_tensor(test)).numpy()
    return train_embedding, test_embedding


def train_autoencoder(
    train: np.ndarray,
    test: np.ndarray,
    config: dict,
    seed: int,
    masked: bool,
):
    set_seed(seed)
    train_z, test_z = standardized_profiles(train, test)
    model = ActivityAutoencoder(int(config["dimension"]), int(config["hidden_dimension"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_z)),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history = []
    for epoch in range(int(config["epochs"])):
        model.train()
        losses = []
        for batch_number, (batch,) in enumerate(loader):
            generator = torch.Generator().manual_seed(seed + epoch * 1000 + batch_number)
            if masked:
                model_input, mask = corrupt_batch(batch, config, generator)
            else:
                model_input = batch
                mask = torch.ones_like(batch, dtype=torch.bool)
            optimizer.zero_grad()
            reconstruction, _ = model(model_input)
            masked_loss = nn.functional.mse_loss(reconstruction[mask], batch[mask])
            loss = masked_loss + (0.05 * nn.functional.mse_loss(reconstruction, batch) if masked else 0.0)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    train_embedding, test_embedding = encode(model, train_z, test_z)
    return train_embedding, test_embedding, model, train_z, test_z, history


def train_contrastive(
    train: np.ndarray,
    test: np.ndarray,
    config: dict,
    seed: int,
):
    set_seed(seed)
    train_z, test_z = standardized_profiles(train, test)
    model = ContrastiveActivityEncoder(int(config["dimension"]), int(config["hidden_dimension"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_z)),
        batch_size=int(config["contrastive_batch_size"]),
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    temperature = float(config["contrastive_temperature"])
    history = []
    for epoch in range(int(config["epochs"])):
        model.train()
        losses = []
        for batch_number, (batch,) in enumerate(loader):
            generator1 = torch.Generator().manual_seed(seed + epoch * 2000 + batch_number * 2)
            generator2 = torch.Generator().manual_seed(seed + epoch * 2000 + batch_number * 2 + 1)
            view1, _ = corrupt_batch(batch, config, generator1, contrastive=True)
            view2, _ = corrupt_batch(batch, config, generator2, contrastive=True)
            projection1 = nn.functional.normalize(model.projector(model.encoder(view1)), dim=1)
            projection2 = nn.functional.normalize(model.projector(model.encoder(view2)), dim=1)
            logits = projection1 @ projection2.T / temperature
            labels = torch.arange(len(batch))
            loss = 0.5 * (
                nn.functional.cross_entropy(logits, labels)
                + nn.functional.cross_entropy(logits.T, labels)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    train_embedding, test_embedding = encode(model, train_z, test_z)
    return train_embedding, test_embedding, history


def breslow_cox_loss(
    eta: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    terms = []
    for point in torch.unique(time[event > 0]):
        deaths = (time == point) & (event > 0)
        risk_set = time >= point
        terms.append(
            -(eta[deaths].sum() - deaths.sum() * torch.logsumexp(eta[risk_set], dim=0))
        )
    return torch.stack(terms).sum() / event.sum().clamp_min(1.0)


def risk_aware_finetune(
    masked_model: ActivityAutoencoder,
    train_z: np.ndarray,
    test_z: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    offset: np.ndarray,
    config: dict,
    seed: int,
):
    set_seed(seed)
    model = copy.deepcopy(masked_model)
    beta = nn.Parameter(torch.zeros(int(config["dimension"])))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [beta],
        lr=float(config["learning_rate"]) * 0.5,
        weight_decay=float(config["weight_decay"]),
    )
    values = torch.as_tensor(train_z)
    time_tensor = torch.as_tensor(time, dtype=torch.float32)
    event_tensor = torch.as_tensor(event, dtype=torch.float32)
    offset_tensor = torch.as_tensor(offset, dtype=torch.float32)
    history = []
    for epoch in range(int(config["risk_finetune_epochs"])):
        generator = torch.Generator().manual_seed(seed + epoch)
        corrupted, mask = corrupt_batch(values, config, generator)
        reconstruction, _ = model(corrupted)
        embedding = model.encoder(values)
        embedding_z = (embedding - embedding.mean(0)) / embedding.std(0).clamp_min(1e-6)
        eta = offset_tensor + embedding_z @ beta
        reconstruction_loss = nn.functional.mse_loss(reconstruction[mask], values[mask])
        survival_loss = breslow_cox_loss(eta, time_tensor, event_tensor)
        ridge = 0.5 * float(config["risk_beta_ridge"]) * (beta @ beta)
        loss = reconstruction_loss + float(config["risk_loss_weight"]) * survival_loss + ridge
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.detach()),
                "reconstruction_loss": float(reconstruction_loss.detach()),
                "survival_loss": float(survival_loss.detach()),
            }
        )
    return (*encode(model, train_z, test_z), history)
