import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..metric import get_auc


class AutotaggingRunner(LightningModule):
    def __init__(self, model: nn.Module, runner_config: DictConfig):
        super().__init__()
        self.model = model
        self.criterion = nn.BCELoss()
        self.hparams.update(runner_config.optimizer.params)
        self.hparams.update(runner_config.scheduler.params)
        self.hparams.update(runner_config.trainer.params)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=self.hparams.nesterov,
        )

        scheduler = ReduceLROnPlateau(
            optimizer=opt,
            factor=self.hparams.factor,
            patience=self.hparams.patience,
            verbose=self.hparams.verbose,
            mode=self.hparams.mode,
        )

        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        audio, label = batch
        prediction = self.model(audio)
        loss = self.criterion(prediction, label)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, label = batch
        prediction = self.model(audio)
        loss = self.criterion(prediction, label)
        return {"val_loss": loss, "predictions": prediction, "labels": label}

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        predctions = torch.stack([output["predictions"] for output in outputs])
        labels = torch.stack([output["labels"] for output in outputs])
        roc_auc, pr_auc = get_auc(predctions, labels)
        self.log_dict(
            {
                "val_loss": val_loss,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {"val_loss": val_loss, "roc_auc": roc_auc, "pr_auc": pr_auc}

    def test_step(self, batch, batch_idx):
        audio, label = batch
        prediction = self.model(audio)
        loss = self.criterion(prediction, label)
        return {"val_loss": loss, "predictions": prediction, "labels": label}

    def test_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        predctions = torch.stack([output["predictions"] for output in outputs])
        labels = torch.stack([output["labels"] for output in outputs])
        roc_auc, pr_auc = get_auc(predctions, labels)
        return {"val_loss": val_loss, "roc_auc": roc_auc, "pr_auc": pr_auc}
