import torch 
from itertools import pairwise


class SinusoidalPositionalEmbedding(torch.nn.Module):

    def __init__(self, size: int = 32, base: float = 1e4) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(base**torch.arange(0, -1, -2 / size)[None])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        wt = self.w * t[:, None]
        return torch.stack((wt.sin(), wt.cos()), 2).flatten(1)

class SimpleCondNet(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, input_dim, condition_dim, output_dim, use_activation=True):
            super().__init__()
            self.use_activation = use_activation
            self.x_encoder = torch.nn.Linear(input_dim, output_dim)
            self.c_encoder = torch.nn.Sequential(
                torch.nn.Linear(condition_dim, output_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(output_dim, output_dim)
            )
        
        def forward(self, x, c):
            x = self.x_encoder(x)
            c = self.c_encoder(c)
            return torch.nn.functional.relu(x + c) if self.use_activation else x + c

    def __init__(self, input_dim, condition_dim, condition_emb_dim, inner_dim: list[int], condition_encoder_dim: list[int] | None=None, **kwargs):
        super().__init__()
        condition_encoder_dim = condition_encoder_dim or []

        self.time_encoder = SinusoidalPositionalEmbedding(condition_emb_dim)

        self.condition_encoder = torch.nn.Sequential()
        for in_dim, out_dim in pairwise([condition_dim] + condition_encoder_dim + [condition_emb_dim]):
            self.condition_encoder.append(torch.nn.Linear(in_dim, out_dim))
            self.condition_encoder.append(torch.nn.ReLU())

        self.blocks = torch.nn.ModuleList([
            SimpleCondNet.Block(in_dim, condition_emb_dim, out_dim) 
            for in_dim, out_dim in pairwise([input_dim] + inner_dim)
        ])
        self.blocks.append(SimpleCondNet.Block(inner_dim[-1], condition_emb_dim, input_dim, use_activation=False))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.condition_encoder(y.float()) + self.time_encoder(t.float())
        for block in self.blocks:
            x = block(x, c)
        return x
    

class SimpleNet(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, input_dim, condition_dim, output_dim, use_activation=True):
            super().__init__()
            self.use_activation = use_activation
            self.x_encoder = torch.nn.Linear(input_dim, output_dim)
            self.c_encoder = torch.nn.Sequential(
                torch.nn.Linear(condition_dim, output_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(output_dim, output_dim)
            )
        
        def forward(self, x, c):
            x = self.x_encoder(x)
            c = self.c_encoder(c)
            return torch.nn.functional.relu(x + c) if self.use_activation else x + c

    def __init__(self, input_dim, condition_emb_dim, inner_dim: list[int], **kwargs):
        super().__init__()
        self.time_encoder = SinusoidalPositionalEmbedding(condition_emb_dim)
        self.blocks = torch.nn.ModuleList([
            SimpleCondNet.Block(in_dim, condition_emb_dim, out_dim) 
            for in_dim, out_dim in pairwise([input_dim] + inner_dim)
        ])
        self.blocks.append(SimpleCondNet.Block(inner_dim[-1], condition_emb_dim, input_dim, use_activation=False))

    def forward(self, x: torch.Tensor, t: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        c = self.time_encoder(t.float())
        for block in self.blocks:
            x = block(x, c)
        return x
    

class CosineScheduler:
    clip_max_value = torch.Tensor([0.999])

    def __init__(self, T: int, s: float = 0.008, beta_min=0.0001, beta_max=0.02):
        """
        Cosine variance scheduler.
        The equation for the variance is:
            alpha_hat = min(cos((t / T + s) / (1 + s) * pi / 2)^2, 0.999)
        The equation for the beta is:
            beta = 1 - (alpha_hat(t) / alpha_hat(t - 1))
        The equation for the beta_hat is:
            beta_hat = (1 - alpha_hat(t - 1)) / (1 - alpha_hat(t)) * beta(t)
        """
        self.T = T
        self._alpha_hats = self._alpha_hat_function(torch.arange(self.T), T, s)
        self._alpha_hats_t_minus_1 = torch.roll(self._alpha_hats, shifts=1, dims=0) # shift forward by 1 so that alpha_first[t] = alpha[t-1]
        self._alpha_hats_t_minus_1[0] = self._alpha_hats_t_minus_1[1]  # to remove first NaN value
        self._betas = 1.0 - self._alpha_hats / self._alpha_hats_t_minus_1
        
        self._betas = torch.clip(self._betas, beta_min, beta_max)
        self._alphas = 1.0 - self._betas
        self._alpha_hats = torch.cumprod(self._alphas, dim=0)
        self._betas_hat = (1 - self._alpha_hats_t_minus_1) / (1 - self._alpha_hats) * self._betas
        self._betas_hat[torch.isnan(self._betas_hat)] = 0.0

    def _alpha_hat_function(self, t: torch.Tensor, T: int, s: float):
        """
        Compute the alpha_hat value for a given t value.
        :param t: the t value
        :param T: the total amount of noising steps
        :param s: smoothing parameter
        """
        cos_value = torch.pow(torch.cos((t / T + s) / (1 + s) * torch.pi / 2.0), 2)
        return cos_value

    def get_alpha_hat(self):
        return self._alpha_hats

    def get_alphas(self):
        return self._alphas

    def get_betas(self):
        return self._betas

    def get_betas_hat(self):
        return self._betas_hat
    

from typing import Any
import pytorch_lightning as pl
from dataclasses import dataclass, asdict
import tqdm 


@dataclass
class NetworkConfig:
    input_dim: int
    condition_emb_dim: int
    inner_dim: list[int]


class DDPM(pl.LightningModule):
    def __init__(self, network_config: NetworkConfig):
        super().__init__()
        self.save_hyperparameters()
        self.network_config = network_config
        self.net = SimpleNet(**asdict(network_config))

        # forward process parameters 
        self.diffusion_steps = 1000
        self.variance_scheduler = CosineScheduler(1000, 0.00001, 0.000001, 0.99)
        self.betas = self.variance_scheduler.get_betas().to(self.device)
        self.alphas = self.variance_scheduler.get_alphas().to(self.device)
        self.alphas_hat = self.variance_scheduler.get_alpha_hat().to(self.device)

    def _params_to_device(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_hat = self.alphas_hat.to(device)
    
    def on_train_start(self) -> None:
        self._params_to_device(self.device)

    def forward(self, x, t, y=None):
        return self.net(x, t, y)
    
    def forward_diffusion(self, x_0, t, eps=None):
        """q(x_t | x_0)"""
        eps = torch.randn_like(x_0) if eps is None else eps
        alpha_hat_t = self.alphas_hat[t].view(-1, *([1] * (x_0.dim() - 1)))
        return torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * eps

    def _common_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        batch_size = batch[0].size(0)
        device = self.device

        X_0, Y = batch
        t = torch.randint(0, self.diffusion_steps - 1, (batch_size, ), device=device)
        noise = torch.randn_like(X_0, device=device)
        X_T = self.forward_diffusion(X_0, t, noise)
        noise_pred = self.forward(X_T, t, Y)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        return loss
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self._common_step(batch, batch_idx)
        self.log('training_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        if self.current_epoch == 0:
            self._params_to_device(self.device)
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=1e-4)
    
    @torch.no_grad()
    def sample(self, num_samples: int, return_intemediates: bool = False, show_progress: bool = False, Y: torch.Tensor | None = None):
        device = self.device
        self._params_to_device(device)
        if Y is not None:
            assert Y.size(0) == num_samples

        X_T = torch.randn(num_samples, self.network_config.input_dim, device=device)
        intermediates = []
        sigmas = torch.sqrt(self.betas)
        for t in tqdm.tqdm(range(self.diffusion_steps - 1, -1, -1), total=self.diffusion_steps, desc="Sampling", disable=not show_progress):
            noise_pred = self.forward(X_T, torch.tensor([t], device=device), Y)
            alpha = self.alphas[t]
            alpha_hat = self.alphas_hat[t]
            
            X_T_denoised_tmp = (1 / torch.sqrt(alpha)) * \
                (X_T - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * noise_pred)
            X_T = X_T_denoised_tmp if t == 0 else X_T_denoised_tmp + sigmas[t] * torch.randn_like(X_T)
            intermediates.append(X_T)
        return torch.stack(intermediates) if return_intemediates else X_T.unsqueeze(0)
    

@dataclass 
class ConditionalNetworkConfig(NetworkConfig):
    condition_encoder_dim: list[int]
    condition_dim: int


class ConditionalDDPM(DDPM):
    def __init__(self, network_config: ConditionalNetworkConfig):
        super().__init__(network_config)
        self.net = SimpleCondNet(**asdict(network_config))

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=1e-4)
    
