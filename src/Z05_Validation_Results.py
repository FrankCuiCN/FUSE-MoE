import os
import pandas as pd
import wandb

api = wandb.Api()

runs = {
    "B01": "wandb link here",
    "B02": "wandb link here",
    "B03": "wandb link here",
    "B04": "wandb link here",
    "P01": "wandb link here",
    "P02": "wandb link here",
    "P03": "wandb link here",
    "M01": "wandb link here",
    "M02": "wandb link here",
    "M03": "wandb link here",
    "M04": "wandb link here",
    "U01": "wandb link here",
    "U02": "wandb link here",
    "U03": "wandb link here",
    "U04": "wandb link here",
    "G01": "wandb link here",
    "G02": "wandb link here",
    "G03": "wandb link here",
    "E05": "wandb link here",
    "A06": "wandb link here",
    "A08": "wandb link here",
}

os.makedirs("Validation Results", exist_ok=True)

for tag, path in runs.items():
    try:
        run = api.run(path)
        # Fetch the full history but only keep the loss_val column.
        hist = run.history(keys=["loss_val"])
        loss_series = hist["loss_val"].dropna()
        if loss_series.empty:
            print(f"{tag}: no loss_val found.")
            continue
        loss_series.to_csv(f"Validation Results/{tag}.csv", index=False, header=["loss_val"])
        print(f"{tag}: saved.")
    except Exception as e:
        print(f"{tag}: failed ({e}).")
