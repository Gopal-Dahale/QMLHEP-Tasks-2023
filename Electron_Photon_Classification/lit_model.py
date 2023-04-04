from model import GraphGNNModel
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import AUROC
import torch.optim as optim

class GraphLevelGNN(pl.LightningModule):

    def __init__(self, batch_size, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.task = 'binary'
        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()
        self.train_auc = AUROC(task=self.task)
        self.val_auc = AUROC(task=self.task)
        self.test_auc = AUROC(task=self.task)
        self.batch_size = batch_size

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch, mode="train")
        loss = self.loss_module(logits, batch.y.float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.train_auc(logits, batch.y)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True,batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch, mode="val")
        loss = self.loss_module(logits, batch.y.float())
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        self.val_auc(logits, batch.y)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch, mode="val")
        self.test_auc(logits, batch.y)
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True, batch_size=self.batch_size)