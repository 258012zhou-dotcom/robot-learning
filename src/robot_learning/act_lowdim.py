"""Teaching-scale ACT-style CVAE/Transformer for low-dimensional observations.

This deliberately omits ACT's multi-camera vision backbone and is not a paper
reproduction. It isolates the training-posterior versus inference-prior path.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ACTLowDimOutput:
    actions: Tensor
    posterior_mean: Tensor | None
    posterior_logvar: Tensor | None


class ACTLowDim(nn.Module):
    """Encode demonstration chunks into z, then decode H bounded actions."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        horizon: int,
        *,
        hidden_size: int = 32,
        latent_size: int = 8,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        for name, value in (
            ("observation_size", observation_size),
            ("action_size", action_size),
            ("horizon", horizon),
            ("hidden_size", hidden_size),
            ("latent_size", latent_size),
            ("attention_heads", attention_heads),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_size % attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads")

        self.observation_size = observation_size
        self.action_size = action_size
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.attention_heads = attention_heads

        self.observation_projection = nn.Linear(observation_size, hidden_size)
        self.action_projection = nn.Linear(action_size, hidden_size)
        self.posterior_position = nn.Parameter(
            torch.randn(horizon + 1, hidden_size) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.posterior_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1, enable_nested_tensor=False
        )
        self.posterior_projection = nn.Linear(hidden_size, latent_size * 2)
        self.latent_projection = nn.Linear(latent_size, hidden_size)
        self.action_queries = nn.Parameter(torch.randn(horizon, hidden_size) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.action_head = nn.Linear(hidden_size, action_size)

    def forward(
        self,
        observations: Tensor,
        target_actions: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> ACTLowDimOutput:
        if (
            observations.ndim != 2
            or observations.shape[1] != self.observation_size
        ):
            raise ValueError("observations have the wrong shape")
        if (target_actions is None) != (valid_mask is None):
            raise ValueError("target_actions and valid_mask must be given together")

        state_token = self.observation_projection(observations)
        posterior_mean: Tensor | None = None
        posterior_logvar: Tensor | None = None
        if target_actions is not None and valid_mask is not None:
            expected_actions = (
                observations.shape[0], self.horizon, self.action_size
            )
            if target_actions.shape != expected_actions:
                raise ValueError("target_actions have the wrong shape")
            if (
                valid_mask.shape != expected_actions[:2]
                or valid_mask.dtype != torch.bool
            ):
                raise ValueError("valid_mask has the wrong shape or dtype")
            if not torch.all(valid_mask[:, 0]):
                raise ValueError("every training chunk needs its first action")

            action_tokens = self.action_projection(target_actions)
            encoder_tokens = torch.cat(
                [state_token[:, None, :], action_tokens], dim=1
            ) + self.posterior_position[None, :, :]
            # The state token is always valid; padding actions cannot be attended to.
            padding_mask = torch.cat(
                [
                    torch.zeros(
                        (observations.shape[0], 1),
                        dtype=torch.bool,
                        device=observations.device,
                    ),
                    ~valid_mask,
                ],
                dim=1,
            )
            encoded = self.posterior_encoder(
                encoder_tokens, src_key_padding_mask=padding_mask
            )
            posterior_mean, posterior_logvar = self.posterior_projection(
                encoded[:, 0, :]
            ).chunk(2, dim=-1)
            standard_noise = torch.randn_like(posterior_mean)
            latent = posterior_mean + standard_noise * torch.exp(
                0.5 * posterior_logvar
            )
        else:
            # The target chunk is unavailable at execution time.
            latent = observations.new_zeros(
                (observations.shape[0], self.latent_size)
            )

        return ACTLowDimOutput(
            actions=self._decode(state_token, latent),
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
        )

    def predict_with_latent(self, observations: Tensor, latent: Tensor) -> Tensor:
        """Diagnostic-only inference with a caller-specified style variable."""
        if observations.ndim != 2 or observations.shape[1] != self.observation_size:
            raise ValueError("observations have the wrong shape")
        if latent.shape != (observations.shape[0], self.latent_size):
            raise ValueError("latent has the wrong shape")
        return self._decode(self.observation_projection(observations), latent)

    def _decode(self, state_token: Tensor, latent: Tensor) -> Tensor:
        memory = torch.stack([state_token, self.latent_projection(latent)], dim=1)
        queries = self.action_queries[None, :, :].expand(
            state_token.shape[0], -1, -1
        )
        decoded = self.action_decoder(queries, memory)
        return torch.tanh(self.action_head(decoded))


def act_lowdim_loss(
    output: ACTLowDimOutput,
    target_actions: Tensor,
    valid_mask: Tensor,
    *,
    kl_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return total loss, valid-slot L1, and KL to a unit Gaussian prior."""
    if not 0.0 <= kl_weight < float("inf"):
        raise ValueError("kl_weight must be finite and non-negative")
    if output.posterior_mean is None or output.posterior_logvar is None:
        raise ValueError("loss requires a training-time posterior")
    if output.actions.shape != target_actions.shape or output.actions.ndim != 3:
        raise ValueError("predicted and target action chunks must match")
    if valid_mask.shape != target_actions.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must match action chunks")
    valid_count = valid_mask.sum()
    if not torch.any(valid_mask):
        raise ValueError("loss requires at least one valid action")

    absolute_error = (output.actions - target_actions).abs()
    action_l1 = (absolute_error * valid_mask.unsqueeze(-1)).sum() / (
        valid_count * target_actions.shape[2]
    )
    kl = -0.5 * (
        1.0
        + output.posterior_logvar
        - output.posterior_mean.square()
        - output.posterior_logvar.exp()
    ).sum(dim=-1).mean()
    return action_l1 + kl_weight * kl, action_l1, kl
