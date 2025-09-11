import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model_hourly(model, dataloader, device, scaler=None, feature_names=None):
    """
    Evaluate a forecasting model and return predictions/targets aggregated to hourly resolution.

    Steps:
      1. Run the model on all batches from dataloader (no gradient updates).
      2. Collect predictions and true targets across all batches.
      3. Collapse any forecast-step dimension .
      4. Inverse-transform results back to original scale using a fitted scaler.
      5. Aggregate results into hourly bins (mean per 60 timesteps).

    Args:
        model: PyTorch model (produces forecasts + latent distribution outputs).
        dataloader: DataLoader providing (inputs, targets).
        device: torch device ('cpu' or 'cuda').
        scaler: (optional) fitted scaler with `.inverse_transform` for rescaling data.
        feature_names: (optional) list of feature names (unused in function, for clarity only).

    Returns:
        preds_hourly: numpy array of hourly-aggregated predictions.
        trues_hourly: numpy array of hourly-aggregated ground truths.
    """
    model.eval()                        
    all_preds, all_targets = [], []      
    # Disabling gradient computation
    with torch.no_grad():                 
        for inputs, targets in dataloader:
            # Reshaping inputs from [B, win, F] → [win, B, F] (expected by our transformer model)
            inputs = inputs.permute(1,0,2).to(device)
            # Forward pass through the model
            outputs, mu, logvar, _ = model(inputs)
            # Saving predictions and targets (convert to numpy on CPU)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # concatenate all batches into single arrays
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_targets, axis=0)

    # If multi-step forecasts, I am avergaing across different steps 
    if preds.ndim == 3:                   # shape (N, H, F)
        preds = preds.mean(axis=1)        # → (N, F)
    if trues.ndim == 3:
        trues = trues.mean(axis=1)

    # Inverse transforming back to original
    if scaler is not None:
        dummy_cols = scaler.n_features_in_
        preds_full = np.zeros((len(preds), dummy_cols))
        trues_full = np.zeros((len(trues), dummy_cols))

        # Fill first columns with predicted/true features
        preds_full[:, :preds.shape[1]] = preds
        trues_full[:, :trues.shape[1]] = trues

        # Inverse transforming then slice back to feature dimension
        preds = scaler.inverse_transform(preds_full)[:, :preds.shape[1]]
        trues = scaler.inverse_transform(trues_full)[:, :trues.shape[1]]

    # Aggregating into hourly means
    def aggregate_hourly(data, interval=60):
        n = len(data) // interval       
        return data[:n*interval].reshape(n, interval, -1).mean(axis=1)

    # Applying hourly aggregation to both preds and trues
    preds_hourly = aggregate_hourly(preds)
    trues_hourly = aggregate_hourly(trues)

    return preds_hourly, trues_hourly

def evaluate_individual_hourly_forecast(
    model, dataloader, device, *,
    scaler, feature_names, scaler_feature_order=None,
    horizon_reduce="mean", sampling_interval=60,
    aggregate_to_minutes=None, plot_n=500
):
    """
    Evaluate a forecasting model at the individual feature level, inverse-transform results,
    optionally aggregate to larger time bins, and plot forecasts vs ground truth.

    Steps:
      1. Run the model on all batches, collect predictions and targets.
      2. Handle multi-step forecasts: reduce across horizon dimension
         (options: 'mean', 'first', 'last').
      3. Inverse transform results back to original feature scale using scaler.
      4. (Optional) Aggregate predictions/targets to coarser temporal resolution
         (e.g., hourly from minute-level).
      5. Plot predictions vs actual values for each selected feature.

    Args:
        model: PyTorch model producing forecasts.
        dataloader: DataLoader yielding (inputs, targets).
        device: torch device ("cpu" or "cuda").
        scaler: fitted sklearn scaler used to normalize training data.
        feature_names: list of feature names to evaluate/plot.
        scaler_feature_order: optional, explicit order of features used to fit scaler
                              (if scaler doesn’t have `.feature_names_in_`).
        horizon_reduce: how to reduce multi-step outputs:
                        - "mean": average across forecast horizons
                        - "first": keep only first horizon step
                        - "last": keep only last horizon step
        sampling_interval: base interval of the dataset (e.g., 60 = 1 hour per step).
        aggregate_to_minutes: if set, aggregate forecasts/targets to this interval
                              (e.g., from 1h → 3h).
        plot_n: maximum number of points to plot per feature.

    Returns:
        preds_inv: numpy array of inverse-transformed predictions (possibly aggregated).
        trues_inv: numpy array of inverse-transformed ground truth values (same shape).
    """
    import numpy as np, pandas as pd, matplotlib.pyplot as plt

    model.eval()                          # put model in eval mode (no dropout/bn updates)
    preds_list, trues_list = [], []       # storage for batch-level predictions/targets

    with torch.no_grad():                 # disable gradients for evaluation
        for X, Y in dataloader:
            # Rearrange input to [win, B, F] for transformer-style models
            X = X.permute(1,0,2).to(device)

            # Forward pass → predictions
            out, *_ = model(X)

            # Move preds/targets back to CPU + numpy
            p = out.detach().cpu().numpy()
            t = Y.detach().cpu().numpy()

            # Ensure both preds and targets are 3D: [N, H, F]
            if p.ndim == 2: p = p[:, None, :]
            if t.ndim == 2: t = t[:, None, :]

            preds_list.append(p)
            trues_list.append(t)

    # Concatenate all batches → full dataset
    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)

    # --- Horizon reduction (if multi-step forecasts) ---
    if horizon_reduce == "mean":
        preds, trues = preds.mean(1), trues.mean(1)
    elif horizon_reduce == "first":
        preds, trues = preds[:,0,:], trues[:,0,:]
    elif horizon_reduce == "last":
        preds, trues = preds[:,-1,:], trues[:,-1,:]
    else:
        raise ValueError("horizon_reduce must be 'mean'|'first'|'last'")

    # --- Prepare feature ordering for inverse scaling ---
    all_cols = getattr(scaler, "feature_names_in_", None)
    if all_cols is None:
        if scaler_feature_order is None:
            raise ValueError("Provide scaler_feature_order or fit scaler with feature names.")
        all_cols = list(scaler_feature_order)
    else:
        all_cols = list(all_cols)

    names = list(feature_names)   # the subset of features we actually want to evaluate

    # --- Create dummy DataFrames to match scaler input shape ---
    dfP = pd.DataFrame(0.0, index=np.arange(len(preds)), columns=all_cols)
    dfT = pd.DataFrame(0.0, index=np.arange(len(trues)), columns=all_cols)
    dfP[names] = preds
    dfT[names] = trues

    # Inverse transform back to original scale
    invP = scaler.inverse_transform(dfP.values)
    invT = scaler.inverse_transform(dfT.values)

    # Extract only the columns corresponding to our selected features
    idx = [all_cols.index(c) for c in names]
    preds_inv = invP[:, idx]
    trues_inv = invT[:, idx]

    # --- Optional aggregation to coarser time resolution ---
    if aggregate_to_minutes is not None and aggregate_to_minutes > sampling_interval:
        r = aggregate_to_minutes // sampling_interval   # number of base steps per new bin
        n = len(preds_inv) // r                         # truncate incomplete bins
        preds_inv = preds_inv[:n*r].reshape(n, r, -1).mean(1)
        trues_inv = trues_inv[:n*r].reshape(n, r, -1).mean(1)

    # --- Plot predictions vs targets ---
    end = min(plot_n, len(preds_inv))
    x = np.arange(end)
    for i, name in enumerate(names):
        plt.figure(figsize=(12,4))
        plt.plot(x, trues_inv[:end, i], label="Actual", linewidth=2)
        plt.plot(x, preds_inv[:end, i], label="Predicted", linestyle="--", linewidth=2)
        plt.title(f"Forecast — {name}")
        plt.xlabel("Time Step hourly")
        plt.ylabel(name)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return preds_inv, trues_inv


