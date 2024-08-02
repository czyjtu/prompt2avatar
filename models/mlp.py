from pathlib import Path
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import importlib
import models.callbacks as callbacks
importlib.reload(callbacks)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch 
from models.callbacks import SimpleRegressionCallbacks
from models.diffusion.ddpm import ConditionalDDPM, SimpleCondNet, ConditionalNetworkConfig
import torch.utils.data 
import wandb 

class DNARegressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes: list[int]):
        super(DNARegressor, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
    

class MLP(pl.LightningModule):
    def __init__(self, model: nn.Module, lr:float=1e-3):
        super(MLP, self).__init__()
        self.model = model 
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)#torch.optim.SGD(self.parameters(), lr=self.lr) #
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
def train_mlp(X_train, X_test, Y_train, Y_test, test_images, EXPERIMENT_DIR: Path, GENESET: str, feature_names: str, lr: float = 1e-3, force: bool = False, model_raw: nn.Module = None, model_raw_name: str = None):
    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train)
    X_train_sc = scaler_x.transform(X_train)
    X_test_sc = scaler_x.transform(X_test)
    
    X_th = torch.tensor(X_train_sc).float()
    Y_th = torch.tensor(Y_train).float()

    train_dataset = torch.utils.data.TensorDataset(Y_th, X_th)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(torch.tensor(Y_test).float(), torch.tensor(X_test_sc).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


    # Initialize the model
    input_dim = Y_train.shape[1]
    output_dim = X_train.shape[1]

    mlp_model_path = EXPERIMENT_DIR / 'mlp_model.ckpt'
    if mlp_model_path.exists() and not force:
        # model_raw = DNARegressor(input_dim, output_dim, [256])
        model = MLP.load_from_checkpoint(mlp_model_path)
    else:
        mlp_model_path.parent.mkdir(parents=True, exist_ok=True)
        model_raw = model_raw or (input_dim, output_dim, [512, 512, 512])
        model_name = model_raw_name or "mlp"
        model = MLP(model_raw)

        import wandb 

        with wandb.init(
            project="prompt2avatar", entity="czyjtu",
            tags=["mlp", model_name, GENESET, feature_names], group="mlp" + GENESET, mode="disabled"
            ) as wandb_run:
            # Initialize Wandb logger
            wandb_logger = WandbLogger(project='prompt2avatar', entity="czyjtu", log_model=True)

            # Initialize Lightning Trainer with checkpointing and logging
            checkpoint_callback = pl_callbacks.ModelCheckpoint(
                monitor='val_loss',
                dirpath='checkpoints',
                filename='best_model',
                save_top_k=1,
                mode='min'
            )
            callback = SimpleRegressionCallbacks(
                        torch.tensor(test_images).float(),
                        torch.tensor(X_test_sc).float(), 
                        torch.tensor(Y_test).float(), 
                        scaler_x
                    )

            trainer = pl.Trainer(
                max_epochs=60,
                logger=wandb_logger,
                callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=100), callback],
                default_root_dir=EXPERIMENT_DIR / "training_logs",
                enable_checkpointing=True,
            )  # You can adjust max_epochs and other parameters

            # Train the model
            trainer.fit(model, train_loader, test_loader)
        
        best_model_path = checkpoint_callback.best_model_path
        # move this file to the experiment directory
        best_model_path = Path(best_model_path)
        best_model_path.rename(mlp_model_path)
    return model