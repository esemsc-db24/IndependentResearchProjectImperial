import torch
from torch.utils.data import Dataset

# ============================== DEFINING CLASS DATASET TO HANDLE THE DATA ============================
class SlidingWindowDataset(Dataset):
    """
    PyTorch Dataset for creating sliding windows from multivariate time series data.

    This class slices a sequence into fixed-length input windows (past observations) 
    and their corresponding forecast targets (future observations). It is useful 
    when preparing sequential data for forecasting tasks.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        2D array-like of shape (n_samples, n_features). Time-series data.
    window_size : int, default=8
        Number of past timesteps to include in each input window.
    forecast_steps : int, default=1
        Number of future timesteps to predict.
    step : int, default=1
        Step size (stride) between the start of consecutive windows.

    Attributes
    ----------
    windows : list of np.ndarray
        List of input windows, each of shape (window_size, n_features).
    targets : list of np.ndarray
        List of prediction targets, each of shape (forecast_steps, n_features).
    """
    def __init__(self, data, window_size=8, forecast_steps=1, step=1):
        ## Storing parameters
        self.window_size = window_size
        self.forecast_steps = forecast_steps
        self.step = step

        # Create sliding windows - how many sliding windows can be extracted?
        n_samples, n_features = data.shape
        num_windows = (n_samples - window_size - forecast_steps) // step + 1

        self.windows = []
        self.targets = []

        ## Looping to generate all possible windows
        ## Iterate through each window and select window as input
        ## target as what to predict
        for i in range(num_windows):
            start = i * step
            window = data[start:start + window_size]
            target = data[start + window_size:start + window_size + forecast_steps]

            self.windows.append(window)
            self.targets.append(target)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = torch.FloatTensor(self.windows[idx])
        target = torch.FloatTensor(self.targets[idx])

        # For single step prediction, squeeze the target
        ## This is because at the start I was doing single step prediction
        if self.forecast_steps == 1:
            target = target.squeeze(0)

        return window, target

class PatientLatentDataset(Dataset):
    """
    Pytorch Dataset for creating sliding windows of scaled features with aligned metadata.
    This dataset slices a multivariate time-series into fixed-length 
    input windows and pairs each window with metadata

    Parameters
    ----------
    raw_data : pd.DataFrame
        Patient-level time-series data containing both features 
        and metadata columns.
    feature_cols : list of str
        Column names to use as input features for the model.
    scaler : sklearn-like transformer
        Including scaler used
    window_size : int, default=8
        Number of past timesteps included in each input window.
    forecast_steps : int, default=10
        Forecast horizon in timesteps. Metadata is aligned with 
        the future row at (i + window_size + forecast_steps - 1).
    
    Attributes
    ----------
    windows : list of np.ndarray
        List of input windows, each shaped (window_size, len(feature_cols)).
    metas : list of dict
        List of dictionaries containing patient ID, pollution measures, 
        and physiological variables aligned with the forecast horizon.
    """
    def __init__(self, raw_data, feature_cols, scaler, window_size=8, forecast_steps=10):
        self.data = raw_data.reset_index(drop=True)
        self.features = scaler.transform(self.data[feature_cols])
        self.window_size = window_size
        self.forecast_steps = forecast_steps

        self.windows = []
        self.metas = []

        # max index ensures alignment with forecast_steps
        max_i = len(self.data) - window_size - forecast_steps + 1
        for i in range(max_i):
            x = self.features[i : i + window_size]

            # future index respects forecast_steps (same as notebook version)
            future_idx = i + window_size + forecast_steps - 1
            future     = self.data.iloc[future_idx]

            self.windows.append(x)
            self.metas.append({
                'patient_id': future['patient_id'],
                'pm2_5_x': future['pm2_5_x'],
                'pm10': future['pm10'],
                'no': future['no'],
                'no2': future['no2'],
                'o3': future['o3'],
                'so2': future['so2'],
                'co': future['co'],
                'inhale_tv': future['inhale_tv'],
                'br_avg': future['br_avg'],
                'br_std': future['br_std'],
                'act_level': future['act_level'],
                'step_count': future['step_count'],
                'temperature': future['temperature'],
                'humidity': future['humidity']
            })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]), self.metas[idx]




