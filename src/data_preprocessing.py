# Data preprocessing functions for sequence creation
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_and_prepare(filepath, item_id, seq_length):
    df = pd.read_csv(filepath)
    df['month'] = pd.to_datetime(df['month'])
    df = df[df['item_id'] == item_id]
    df = df.groupby('month')['item_cnt_month'].sum().reset_index()
    df = df.set_index('month').asfreq('MS').fillna(0)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return create_sequences(scaled_data, seq_length), scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)
