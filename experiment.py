# Import necessary modules and functions
from AND_nn_exp import ANDismabiguator  # Import the ANDismabiguator model
from AND_readdata_exp import (
    PairsANDDataModule2,
)  # Import the data module for loading pairs data
from lightning.pytorch import (
    seed_everything,
)  # Function to set seed for reproducibility
from pytorch_lightning.loggers import (
    CSVLogger,
)  # CSV logger for logging training metrics
import pytorch_lightning as pl  # Import PyTorch Lightning
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)  # Callback for saving model checkpoints
import sys  # Module for accessing system-specific parameters and functions
from dataclasses import (
    dataclass,
)  # Dataclass decorator for creating configuration classes

# Set the seed for reproducibility using the first command-line argument
seed_everything(int(sys.argv[1]))


@dataclass
class Config:
    """Configuration for logger"""

    save_dir: str = "logs/specter_chars2vec"  # Directory to save logs


# Create a config instance
config = Config()

# Initialize the data module with specified batch and chunk sizes
data_module = PairsANDDataModule2(batch_size=2048, chunk_size=500000)

# Set up the data module for the 'fit' stage (training)
data_module.setup(stage="fit")
# Get the training data loader
train_loader = data_module.train_dataloader()
# Get the validation data loader
val_loader = data_module.val_dataloader()


def run_experiment(number):
    """Run the experiment with a given number (for naming/logging purposes)"""

    # Template for checkpoint filenames
    filename_template = f"{number}_specter_chars2vec_" + "{epoch}-{val_loss:.2f}"

    # Create a checkpoint callback to save model checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Monitor the validation loss
        mode="min",  # Save checkpoints with the minimum validation loss
        save_weights_only=True,  # Save only the model weights
        dirpath="/mnt/home/amadovic/neural_author_disambiguator/hyperparametertuning/specter_chars2vec",  # Directory to save checkpoints
        filename=filename_template,  # Template for checkpoint filenames
        save_last=True,  # Save the last checkpoint
        every_n_epochs=1,  # Save checkpoint after every epoch
    )

    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(
        accelerator="gpu",  # Use GPU for training
        devices=1,  # Number of GPUs to use
        max_epochs=30,  # Maximum number of epochs to train
        logger=CSVLogger(
            save_dir=config.save_dir,  # Directory to save logs
            name=f"{number}_specter_chars2vec.log",  # Log file name
        ),
        enable_checkpointing=True,  # Enable checkpointing
        callbacks=[checkpoint_callback],  # Add the checkpoint callback
    )

    # Initialize the model with the given run number
    model = ANDismabiguator(run_num=int(sys.argv[1]))

    # Fit the model using the trainer
    trainer.fit(model, train_loader, val_loader)


# Main entry point of the script
if __name__ == "__main__":
    # Run the experiment with the run number provided as the first command-line argument
    run_experiment(int(sys.argv[1]))
