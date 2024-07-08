import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="classifier_kgc",

    # track hyperparameters and run metadata
    config={
    "class":16,
    "dim":"768_512_256",
    "split":20,
    "layers":3,
    "epochs": 50,
    }
)