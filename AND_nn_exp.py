# Import necessary modules and classes
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryROC
import numpy as np


# Define the ANDismabiguator class, which inherits from PyTorch Lightning's LightningModule
class ANDismabiguator(pl.LightningModule):

    def __init__(self, run_num, dropout_prob=0.3):
        super().__init__()
        self.run_num = run_num

        # Define the layers of the neural network using a Sequential container
        self.l1 = nn.Sequential(
            nn.Linear(868, 1024),  # First linear layer
            nn.LeakyReLU(),  # Activation function
            nn.Dropout(p=dropout_prob),  # Dropout layer
            nn.Linear(1024, 1024),  # Second linear layer
            nn.LeakyReLU(),  # Activation function
            nn.Dropout(p=dropout_prob),  # Dropout layer
            nn.Linear(1024, 1024),  # Third linear layer
            nn.LeakyReLU(),  # Activation function
            nn.Dropout(p=dropout_prob),  # Dropout layer
            nn.Linear(1024, 1024),  # Fourth linear layer
            nn.LeakyReLU(),  # Activation function
            nn.Dropout(p=dropout_prob),  # Dropout layer
            nn.Linear(1024, 1024),  # Fifth linear layer
            nn.LeakyReLU(),  # Activation function
            nn.Dropout(p=dropout_prob),  # Dropout layer
            nn.Linear(1024, 64),  # Final linear layer to reduce dimensions
            nn.LeakyReLU(),  # Activation function
        )

        # Set margin for loss functions
        self.margin = 0.5

        # Initialize metrics and loss functions
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.CosineEmbeddingLoss = nn.CosineEmbeddingLoss(margin=self.margin)
        self.TripletMarginWithDistanceLoss = nn.TripletMarginWithDistanceLoss(
            distance_function=nn.CosineSimilarity(), margin=self.margin
        )

        # Initialize containers for storing outputs
        self.trainoutputs = []
        self.outputs = []
        self.optimal_thresholds = []

    def forward(self, x):
        # Define the forward pass of the network
        return self.l1(x)

    def ANDloss(self, batch):
        # Custom loss function for the AND model
        x, y = batch
        y_hat = self(x)
        sign = torch.where(y, torch.tensor(1), torch.tensor(-1))
        cosine_sim = torch.cosine_similarity(y_hat[:, 0, :], y_hat[:, 1, :])
        loss = self.CosineEmbeddingLoss(y_hat[:, 0, :], y_hat[:, 1, :], sign)
        accuracy = self.accuracy(torch.clamp(cosine_sim, min=0, max=1), y)
        total_loss = loss.mean()
        self.trainoutputs.append({"y": y, "CS": cosine_sim})
        return accuracy, total_loss

    def nceloss(self, batch):
        # Custom loss function using negative sampling
        temperature = 1.0
        x, y = batch
        y_hat = self(x)
        cosine_sim = torch.cosine_similarity(y_hat[:, 0, :], y_hat[:, 1, :])
        positive_indices = torch.where(y == 1)[0]
        cosine_sim = cosine_sim / temperature
        nll = -cosine_sim[positive_indices] + torch.logsumexp(cosine_sim, dim=-1)
        loss = nll.mean()
        accuracy = self.accuracy(torch.clamp(cosine_sim, min=0, max=1), y)
        if self.training:
            self.trainoutputs.append({"y": y, "CS": cosine_sim})
        return accuracy, loss

    def triplet_loss(self, batch):
        # Custom triplet loss function
        x, y = batch
        y_hat = self(x)
        sim = torch.cosine_similarity(y_hat[:, 0, :], y_hat[:, 1, :])
        positive_indices = torch.where(y == 0)[0]
        negative_indices = torch.where(y == 1)[0]
        num_positive = len(positive_indices)
        num_negative = len(negative_indices)
        num_triplets = min(num_positive, num_negative)
        scramble_indices = np.random.randint(0, num_triplets, size=num_triplets)
        anchor_embeddings = y_hat[positive_indices[:num_triplets]][:, 0, :]
        positive_embeddings = y_hat[positive_indices[:num_triplets]][:, 1, :]
        negative_embeddings = y_hat[scramble_indices[:num_triplets]][:, 1, :]
        loss = self.TripletMarginWithDistanceLoss(
            anchor_embeddings, positive_embeddings, negative_embeddings
        )
        accuracy = self.accuracy(torch.clamp(sim, min=0, max=1), y)
        if self.training:
            self.trainoutputs.append({"y": y, "CS": sim})
        return accuracy, loss

    def training_step(self, batch, batch_idx):
        # Training step
        accuracy, loss = self.ANDloss(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        x, y = batch
        y_hat = self(x)
        cosine_sim = torch.cosine_similarity(y_hat[:, 0, :], y_hat[:, 1, :])
        accuracy, loss = self.ANDloss(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        self.outputs.append({"y": y, "CS": cosine_sim})
        return loss

    def on_train_epoch_end(self):
        # Actions at the end of each training epoch
        y = []
        CS = []
        for outputs in self.trainoutputs[-1:]:
            y.append(outputs["y"])
            CS.append(outputs["CS"])
        targets = torch.stack(y).flatten(0, 1)
        preds = torch.stack(CS).flatten(0, 1)
        self.trainoutputs.clear()
        roc = BinaryROC()
        fpr, tpr, thresholds = roc(preds, targets)
        gmeans = torch.sqrt(tpr * (1 - fpr))
        index = torch.argmax(gmeans)
        optimal_threshold = thresholds[index]
        self.log("train_end_acc", tpr[index], prog_bar=True, sync_dist=True)
        self.log(
            "train_end_threshold",
            optimal_threshold.item(),
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        # Actions at the end of each validation epoch
        y = []
        CS = []
        for outputs in self.outputs:
            y.append(outputs["y"])
            CS.append(outputs["CS"])
        # Flatten the lists of tensors into single tensors
        targets = torch.cat(y)
        preds = torch.cat(CS)
        # Ensure the tensors have the correct shape
        targets = targets.view(-1)
        preds = preds.view(-1)
        roc = BinaryROC()
        fpr, tpr, thresholds = roc(preds, torch.tensor(targets))
        gmeans = torch.sqrt(tpr * (1 - fpr))
        index = torch.argmax(gmeans)
        optimal_threshold = thresholds[index]
        self.outputs.clear()
        self.log("val_end_acc", tpr[index], prog_bar=True, sync_dist=True)
        self.log(
            "val_end_threshold", optimal_threshold.item(), prog_bar=True, sync_dist=True
        )
        self.optimal_thresholds.append(optimal_threshold.item())
        np.save(
            "/mnt/home/amadovic/neural_author_disambiguator/hyperparametertuning/specter_chars2vec/"
            + str(self.run_num),
            np.array(self.optimal_thresholds),
        )

    def test_step(self, batch, batch_idx):
        # Test step
        accuracy, loss = self.ANDloss(batch)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("test_acc", accuracy, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Configure the optimizer
        return torch.optim.Adamax(self.parameters(), lr=0.0001)
