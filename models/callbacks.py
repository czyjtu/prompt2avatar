import pytorch_lightning as pl 
import torch 
import numpy as np
import torchmetrics.regression as rm
import wandb 


class MetricsCallback(pl.Callback):
    def __init__(self, images, X, Y, scaler_x):
        super().__init__()
        self.X = X
        self.Y = Y
        self.scaler_x = scaler_x
        self.mae = rm.MeanAbsoluteError()
        self.r2 = rm.R2Score(num_outputs=X.shape[1], )
        self.images = images
        # self.scaler = wandb.Artifact('scaler' + str(wandb.run.id), type='scaler')
        self.dataset_artifact = wandb.Artifact("eval_data_" + str(wandb.run.id), type='dataset')

    def array_to_table(self, X) -> wandb.Table:
        columns = [f"feature_{i}" for i in range(X.shape[1])]
        table = wandb.Table(columns=columns)
        for row in X:
            table.add_data(*row.tolist())
        return table
    
    def images_to_table(self, images) -> wandb.Table:
        table = wandb.Table(columns=["image"])
        for image in images:
            table.add_data(wandb.Image(image))
        return table
    
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch == 0:
            self.dataset_artifact.add(self.array_to_table(self.X.numpy()), name="X")
            self.dataset_artifact.add(self.array_to_table(self.X.numpy()), name="Y")
            self.dataset_artifact.add(self.images_to_table(self.images.numpy()), name="images")
            # self.scaler.add(self.scaler_x, "scaler_object")
            wandb.run.log_artifact(self.dataset_artifact)
            # wandb.run.log_artifact(self.scaler)

        if trainer.current_epoch % 5 == 0:
            self.X = self.X.to(pl_module.device)
            self.Y = self.Y.to(pl_module.device)
            X_pred = self.sample_model(pl_module)

            predictions = wandb.Artifact('predicted_features_' + str(wandb.run.id), type='predictions')
            predictions.add(self.array_to_table(X_pred), f"X_pred_{trainer.current_epoch}")

            self.mae = self.mae.to(pl_module.device)
            self.r2 = self.r2.to(pl_module.device)
            metrics, names = self._get_metrics(X_pred)
            for metr, name in zip(metrics, names):
                pl_module.log(name, metr, prog_bar=True)

            wandb.run.log_artifact(predictions)

    def sample_model(self, pl_module):
        pl_module.eval()
        n_samples = len(self.X)
        samples = pl_module.sample(n_samples, show_progress=False, Y=self.Y, return_intemediates=False)[-1]
        pl_module.train()
        return samples
    
    def _get_metrics(self, samples):
        rmse_loss = torch.sqrt(torch.mean((samples - self.X)**2))
        samples_rescaled = self.scaler_x.inverse_transform(samples.cpu().detach().numpy())
        X_rescaled = self.scaler_x.inverse_transform(self.X.cpu().detach().numpy())
        scaled_rmse_loss = np.sqrt(np.mean((samples_rescaled - X_rescaled)**2))
        mae = self.mae(samples, self.X)
        r2 = self.r2(samples, self.X)
        return [rmse_loss.item(), scaled_rmse_loss.item(), mae.item(), r2.item()], ['RMSE', 'Scaled RMSE', "MAE", "R2"]
