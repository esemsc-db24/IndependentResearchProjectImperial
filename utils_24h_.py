import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def extract_latents_by_condition(model, dataloader, device):
    """
    Run the model in evaluation mode to extract latent representations 
    and aligned metadata from a dataset.

    Args:
        model: Trained variational time-series model with a reparameterize method.
        dataloader: PyTorch DataLoader yielding (window, metadata) pairs.
        device: Torch device ("cpu" or "cuda") for inference.

    Returns:
        z_df (pd.DataFrame): DataFrame of latent vectors (N x latent_dim).
        meta_df (pd.DataFrame): DataFrame of metadata aligned row-by-row with z_df.
    """
    ## Putting the model in evaluation mode
    model.eval()
    ## This will store the latent vectors
    z_list = []
    ## This will store additional data
    meta_list = []
    ## Turning off gradients - I don't need to keep them
    with torch.no_grad():
    ## In dataloader from PatientLatentDataset I get:
    ## a tensor of shape (batch, window_size and n_features)
    ## Metadata including patient_id, pm_2_5 etc
        for x, meta_batch in dataloader:
            # Reformat for transformer (seq_len, batch, features) and move to device
            x = x.permute(1, 0, 2).to(device)
            # Forward pass: only need latent mean and variance
            _, mu, logvar, _ = model(x)
            # Sample latent vector z using reparameterization trick
            z = model.reparameterize(mu, logvar).cpu().numpy()
            z_list.append(z)

            # Converting to store the data in dataloader
            # k is f.i. patient_id
            # v is the context of the data where every row is raw physiological and pollution form of the data
            clean_meta = {}
            for k, v in meta_batch.items():
                clean_meta[k] = np.array(v).tolist()

            # Building DataFrame for this batch and append
            meta_df_batch = pd.DataFrame(clean_meta)
            meta_list.append(meta_df_batch)

    # Concatenate all batches
    z_all = np.concatenate(z_list, axis=0)
    meta_df = pd.concat(meta_list, ignore_index=True)

    # Return two aligned DataFrames
    z_df = pd.DataFrame(z_all)
    return z_df, meta_df

def normalize_risk_vector(vec, risk_indices, scaler):
    """
    Normalize a vector of risk-related features using the same scaling 
    as the model’s training data.

    Args:
        vec (array-like): Input vector of features, either the full feature vector 
                          or already the subset of risk-related features.
        risk_indices (list[int]): Indices of the features considered risk-relevant 
                                  (e.g., br_avg, br_std, pm10, etc.).
        scaler (sklearn scaler): Fitted scaler used during preprocessing, with the 
                                 same feature ordering as the training data.

    Returns:
        np.ndarray: Normalized values of the risk-related features only.
    """
    # If the input vector vec already has the same length as the list of risk feature indices, 
    # then it must already contain only the risk features.
    if len(vec) == len(risk_indices):
        risk_vec = vec  # It's already the risk vector
    else:
        risk_vec = vec[risk_indices] # Otherwise select risk features
    ## Creating a dummy row of zeros with the same length as the training feature space
    dummy = np.zeros((1, scaler.n_features_in_))
    ##  Plainge the risk feature values into their correct columns (risk_indices)
    dummy[0, risk_indices] = risk_vec  
    ## Applying the fitted scaler consinstelty with training data
    dummy_scaled = scaler.transform(dummy)
    ## Extract only the normalised risk features
    return dummy_scaled[0, risk_indices]

def get_risk_from_prediction(pred_vector, baseline_vector, risk_indices, scaler, method="euclidean", green_thresh=0.3, yellow_thresh=1.0, return_distance=False):
    """
    Compare a predicted risk feature vector against a baseline to assign a 
    risk category (GREEN, YELLOW, RED).

    Steps:
        1. Normalize both the prediction and baseline vectors using the same scaler.
        2. Compute the distance between them (currently Euclidean only).
        3. Classify risk based on thresholds:
           - distance < green_thresh  → "GREEN"
           - distance < yellow_thresh → "YELLOW"
           - otherwise                → "RED"
    Args:
        pred_vector (array-like): Predicted values for risk-related features.
        baseline_vector (array-like): Baseline values for risk-related features.
        risk_indices (list[int]): Indices of features considered risk-related.
        scaler (sklearn scaler): Pre-fitted scaler for normalisation.
        method (str): Distance metric (default: 'euclidean').
        green_thresh (float): Distance threshold for GREEN category.
        yellow_thresh (float): Distance threshold for YELLOW category.
        return_distance (bool): If True, return both (risk, distance).

    Returns:
        str or (str, float): Risk category, optionally with computed distance.
    """
    # Normalise the predicted risk features so they are on the same scale as the training data
    pred_norm = normalize_risk_vector(pred_vector, risk_indices, scaler)
    # Pass baseline_vector directly as it already only contains risk features
    base_norm = normalize_risk_vector(baseline_vector, risk_indices, scaler)
    # Computing the distance between the predicted vector and the baseline
    if method == "euclidean":
        dist = np.linalg.norm(pred_norm - base_norm)
    else:
        raise NotImplementedError("Only Euclidean distance supported")
    # Classify the distance into traffic-light risk categories:
    #    - GREEN:  close to baseline (normal behaviour)
    #    - YELLOW: moderate deviation
    #    - RED:    strong deviation (possible risk)
    if dist < green_thresh:
        risk = "GREEN"
    elif dist < yellow_thresh:
        risk = "YELLOW"
    else:
        risk = "RED"
        
    return (risk, dist) if return_distance else risk

def predict_with_model(model, dataloader, device):
    """
    Run the model in evaluation mode and extract predictions aligned with metadata.  
    Handles both 2D and 3D outputs from the model.

    Args:
        model: Trained model (e.g. VAE, Transformer-based predictor).
        dataloader: PyTorch DataLoader yielding (window, metadata) pairs.
        device: Torch device ("cpu" or "cuda") for inference.

    Returns:
        y_pred (np.ndarray): Array of predictions, shape [N, F].
        meta_df (pd.DataFrame): DataFrame of metadata aligned row-by-row with predictions.
    """
    model.eval()
    preds, metas = [], []

    with torch.no_grad():
        for x, meta_batch in dataloader:
            ## Needed for what the transformer wants
            x = x.permute(1,0,2).to(device)
            # Forward pass → forecast
            frc, mu, logvar, _ = model(x)

            # If frc is 3-D, grab the last time-step;
            # if it’s 2-D, it is the that last step.
            if frc.dim() == 3:
                final = frc[:, -1, :]      # shape [B, F]
            elif frc.dim() == 2:
                final = frc              # shape [B, F]
            else:
                raise ValueError(f"Unexpected recon shape {frc.shape}")

            preds.append(final.cpu().numpy())
            metas.append(pd.DataFrame(meta_batch))

    y_pred  = np.vstack(preds)
    meta_df = pd.concat(metas, ignore_index=True)
    return y_pred, meta_df

def scan_individual_risk(model,dataset,scaler,baseline_vector,risk_indices, device,):
    """
    Evaluate risk levels for an individual across all windows in a dataset.

    For each input window:
      1. Run the model to generate forecasts.
      2. Extract the first forecast step.
      3. Inverse-transform predictions back to original feature space.
      4. Select risk-related features only.
      5. Compute distance to the baseline vector.
      6. Convert that distance into a categorical risk label (GREEN / YELLOW / RED).

    Args:
        model: Trained forecasting model.
        dataset: Dataset yielding sliding windows of time-series input.
        scaler: Fitted scaler used for normalization/inverse transform.
        baseline_vector (np.ndarray): Reference vector of "normal" risk features.
        risk_indices (list[int]): Indices of features used to assess risk.
        device: Torch device ("cpu" or "cuda").

    Returns:
        risks (list[str]): Risk categories per window.
        distances (list[float]): Corresponding distances from the baseline.
    """
    model.eval()
    risks = []
    distances = []

    # let's take the normal state for this person
    baseline_norm = normalize_risk_vector(baseline_vector, risk_indices, scaler)
    with torch.no_grad():
        # Let's looop through each window of the dataset
        for i in range(len(dataset)):
            x_window, _ = dataset[i]
              # adding a batch dimension so the modle goes sample by sample
            x = x_window.unsqueeze(1).to(device)

            frc, mu, logvar, _ = model(x)
            # frc is either
            #   • (1, F)  if forecast_steps==1, or
            #   • (1, H, F) otherwise
            if frc.dim() == 3:
                # pick the first horizon
                first_horizon = frc[:, 0, :]   # (1, F)
            else:
                # already one‐step
                first_horizon = frc           # (1, F)

            vec = first_horizon.squeeze(0).cpu().numpy()  # (F,)

            # inverse‐transform only the risk features back to original units
            D = scaler.n_features_in_
            dummy = np.zeros((1, D), dtype=float)
            dummy[0, : vec.shape[0]] = vec
            inv = scaler.inverse_transform(dummy)[0]      # (D,)
            pred_risk_feats = inv[risk_indices]           # (len(risk_indices),)

            # Here I am comparing predicted risk features with the baseline in Euclidean way
            risk_label, dist = get_risk_from_prediction(
                pred_vector    = pred_risk_feats,
                baseline_vector= baseline_vector,
                risk_indices   = risk_indices,
                scaler         = scaler,
                method         = "euclidean",
                return_distance= True
            )

            risks.append(risk_label)
            distances.append(dist)
            # risk is telling me states per window and the distances are the magnitude
    return risks, distances

def scan_risk(model,dataset,scaler,baseline_vector,risk_indices,pollution_index,device):
    """
    Foreach input window:
    1. Slide through each window in `dataset`and run the model
    2. Take all forecast steps 
    3. Inverse-scale them back to original units,
    and compute:
      • deviations: distance from baseline in risk features
      • pollution:  the pollution value at each forecast step

    Handles both 1-step forecasts (output shape 2D) 
    and multi-step forecasts (output shape 3D).
    """
    model.eval()
    deviations   = []
    pollution_lv = []
    # let's take the normal state for this person and normalise
    baseline_norm = normalize_risk_vector(baseline_vector, risk_indices, scaler)

    with torch.no_grad():
        for i in range(len(dataset)):
            x_window, _ = dataset[i]
            # adding batch dimension
            x = x_window.unsqueeze(1).to(device)   # (seq_len, batch=1, features)
            frc, _, _, _ = model(x)

            # frc making it of shape (batch, H, F)
            if frc.ndim == 2:
                # 1-step: (1, F) → (1, 1, F)
                frc = frc.unsqueeze(1)
            # now frc.shape == (1, H, F)
            # Extracting frc for this window
            arr = frc[0].cpu().numpy()  # (H, F)
            # For each forecast horizon, 
            # the loop will reconstruct the full feature vector, inverse-transforms it
            # It will compute deviation from baseline and pollution level.
            for forecast_step in arr:
                # forecast_step.shape == (F,)
                full = np.zeros((scaler.n_features_in_,), dtype=float)
                full[:forecast_step.shape[0]] = forecast_step

                original = scaler.inverse_transform(full.reshape(1, -1))[0]
                # compute distance in risk-indices
                norm     = normalize_risk_vector(original, risk_indices, scaler)
                dist     = np.linalg.norm(norm - baseline_norm)
                pol_val  = original[pollution_index]

                deviations.append(dist)
                pollution_lv.append(pol_val)

    return deviations, pollution_lv
    
def compute_reactivity_score(deviations, pollution_vals):
    """
    Compute how strongly deviations increase under high pollution compared 
    to low pollution.

    Steps:
        1. Compute the 25th and 75th percentiles of pollution values.
        2. Collect deviations corresponding to low (<=25th) and high (>75th) pollution.
        3. Compute average deviation for low and high groups.
        4. Return avg_low, avg_high, their difference (reactivity score),
           and the (low, high) pollution thresholds.

    Args:
        deviations (list/array): Physiological deviation values over time.
        pollution_vals (list/array): Pollution values aligned with deviations.

    Returns:
        tuple: (avg_low, avg_high, avg_high - avg_low, (low, high))
    """
    # Finding low/high pollution thresholds
    low, high = np.percentile(pollution_vals, [25, 75])

    # Collecting deviations for low pollution
    low_dev_list = []
    for d, p in zip(deviations, pollution_vals):
        if p <= low:
            low_dev_list.append(d)

    # Collecting deviations for high pollution
    high_dev_list = []
    for d, p in zip(deviations, pollution_vals):
        if p > high:
            high_dev_list.append(d)

    # Computing averages (handle empty lists safely)
    avg_low = np.mean(low_dev_list) if low_dev_list else 0.0
    avg_high = np.mean(high_dev_list) if high_dev_list else 0.0

    # Returning results
    return avg_low, avg_high, avg_high - avg_low, (low, high)



def predict_pollution_level(combined: pd.DataFrame, multiplier: float, model, feature_cols: list, scaler, window_size: int, forecast_steps: int, pollution_cols: list, physiological_cols: list, device, PatientLatentDataset
) -> pd.DataFrame:
    """
    Runs predictions at a given pollution multiplier.

    Args:
        combined (pd.DataFrame): Full dataset (train + test).
        multiplier (float): Factor by which to scale pollution features.
        model: Trained model for prediction.
        feature_cols (list): Features used for prediction.
        scaler: Fitted scaler for inverse transformation.
        window_size (int): Input sequence length.
        forecast_steps (int): Forecast horizon.
        pollution_cols (list): Columns representing pollution features.
        physiological_cols (list): Physiological features of interest.
        device: Torch device for inference.

    Returns:
        pd.DataFrame: Metadata + predicted physiological responses under scenario.
    """
    # Applying multiplier to pollution features
    modified_data = combined.copy()
    # It will be multiplied by 1, 2, 4, 6 where 1 is the baseline
    # I will only multiply the pollution cols to see how other features react
    modified_data.loc[:, pollution_cols] *= multiplier
    
    # Creating perturbed dataset and running through model
    dataset = PatientLatentDataset(modified_data, feature_cols, scaler, window_size, forecast_steps)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False)
    y_pred_scaled, meta_df = predict_with_model(model, loader, device)

    # Handle predictions (if 3d flatten, keeping only horizon)
    arr = np.asarray(y_pred_scaled)
    if arr.ndim == 3:
        arr = arr[:, -1, :]  
    arr = arr[:, :len(feature_cols)]

    # Inverse scaling to original units
    pred_scaled_df = pd.DataFrame(arr, columns=feature_cols)
    pred_inv_df    = pd.DataFrame(scaler.inverse_transform(pred_scaled_df), columns=feature_cols)

    # Keeping physiological variables only to plot and see perturbation results, rename as *_pred
    df_phys = pred_inv_df[physiological_cols].add_suffix('_pred')

    # Attach metadata + scenario label
    meta_df = meta_df.reset_index(drop=True).copy()
    meta_df['scenario'] = f'{multiplier}x'

    return pd.concat([meta_df, df_phys.reset_index(drop=True)], axis=1)
    
def get_baseline_prediction_vector(model, dataloader, device, scaler, risk_indices):
    """
    Compute a baseline risk vector for an individual from model predictions.
    Uses only the first forecast step from each input window 
    and averages across all windows to create a normal state.
    
    Args:
        model: Trained forecasting model (e.g. VAE, Transformer).
        dataloader: DataLoader yielding input windows + metadata.
        device: Torch device ("cpu" or "cuda") for inference.
        scaler: Fitted scaler for inverse transforming predictions.
        risk_indices: List of column indices for risk-related features.

    Returns:
        baseline_vec (np.ndarray): 1D array of mean risk features 
                                   (length = len(risk_indices)).
    """
    model.eval()
    all_vectors = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.permute(1, 0, 2).to(device)  # (seq_len, batch, feat)
            frc, mu, logvar, _ = model(inputs)
            
            if frc.dim() == 3:
                first_h  = frc[:, 0, :]     # (batch, features)
            else:
                first_h  = frc             # already (batch, features)

            first_np = first_h.cpu().numpy()  # (batch, features)
            B, F     = first_np.shape
            D        = scaler.n_features_in_

            # build dummy array so we can inverse‐scale
            dummy = np.zeros((B, D), dtype=float)
            dummy[:, :F] = first_np

            inv = scaler.inverse_transform(dummy)[:, :F]  # back to original units

            # Selecting only risk related features
            risk_vecs = inv[:, risk_indices]             # shape (batch, len(risk))
            all_vectors.append(risk_vecs)
    # Here I am stacking all the windows and and returning the mean across all winwods as the baseline vector
    all_vectors = np.vstack(all_vectors)  # shape (num_windows, len(risk))
    return all_vectors.mean(axis=0)       # one vector of length len(risk)
