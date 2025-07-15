import torch
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

def load_tft_model():
    # 1. Load your dataframe
    df = pd.read_csv("app/data/asphalt_temp_data.csv")  # or your actual path

    # 2. Define your dataset the same way as during training
    max_encoder_length = 24
    max_prediction_length = 12

    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= df["time_idx"].max() - max_prediction_length],
        time_idx="time_idx",
        target="target_temp",  # <-- replace with your actual target column
        group_ids=["series_id"],  # <-- replace with your actual group ID
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["target_temp"],
        time_varying_known_reals=["time_idx", "hour", "day"],  # customize if needed
        target_normalizer=GroupNormalizer(groups=["series_id"]),
    )

    # 3. Recreate model and load weights
    model = TemporalFusionTransformer.from_dataset(training, learning_rate=0.03, hidden_size=16, attention_head_size=1)
    model.load_state_dict(torch.load("app/data/asphalt_tft_model.pth"))
    model.eval()

    return model, training
