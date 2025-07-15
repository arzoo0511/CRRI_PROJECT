import pandas as pd
from app.model.load_model import load_tft_model

def predict_future():
    model, training = load_tft_model()

    # Build test dataset
    from pytorch_forecasting import TimeSeriesDataSet
    from torch.utils.data import DataLoader

    # Assume you already have the full dataframe used for training
    df = training.data

    # Select last 12 months as test
    max_idx = df["time_idx"].max()
    test = TimeSeriesDataSet.from_dataset(
        training, df, stop_randomization=True, min_prediction_idx=max_idx - 12
    )
    test_loader = test.to_dataloader(train=False, batch_size=64)

    predictions = model.predict(test_loader, return_x=True)
    pred_df = test.dataset_to_prediction(predictions)

    return pred_df, model
