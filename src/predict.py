# Prediction and evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(y_test, y_pred, scaler):
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    return rmse, mae, y_test_inv, y_pred_inv
