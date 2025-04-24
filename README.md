
# 📊 Demand Forecasting Using LSTM

This project demonstrates an end-to-end pipeline for demand forecasting in a retail environment using **LSTM (Long Short-Term Memory)** neural networks. The model predicts future demand based on historical sales data. It includes **EDA**, **model training**, and a **FastAPI-based deployment** for real-time predictions.

## 📂 Project Structure

```
demand_forecasting_lstm/
│
├── data/
│   ├── raw/                 # Original datasets (CSV from Kaggle)
│   ├── processed/           # Aggregated monthly sales data
│
├── notebooks/
│   ├── EDA_Retail_Sales.ipynb         # Exploratory Data Analysis
│   └── Model_LSTM_Retail.ipynb        # LSTM Model Training & Evaluation
│
├── src/
│   ├── data_preprocessing.py          # Data cleaning and sequence creation
│   ├── model.py                       # LSTM model architecture
│   ├── predict.py                     # Model evaluation utilities
│
├── models/
│   ├── lstm_model.h5                  # Trained LSTM model
│   └── scaler.save                    # Scaler used for normalization
│
├── outputs/
│   ├── figures/                       # Plots and visualizations
│   └── results/                       # Metrics and logs
│
├── app/
│   └── main.py                        # FastAPI app for serving predictions
│
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Files to ignore in Git
```

## 🧠 Why LSTM for Demand Forecasting?

- **LSTM** networks are a type of **Recurrent Neural Network (RNN)** designed to capture **long-term dependencies** in sequential data.
- **Retail demand** often shows **seasonal patterns**, **trends**, and **lags** which LSTM can effectively learn.
- Unlike traditional models (ARIMA, etc.), LSTM can handle **non-linear relationships** and **irregular patterns** in time series data.

## 📝 Workflow Overview

1. **Data Source**: [Kaggle - Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)
2. **EDA**:
   - Explore trends, seasonality, and outliers.
   - Aggregate daily sales to monthly.
   - Found in **`notebooks/EDA_Retail_Sales.ipynb`**.

3. **Model Development**:
   - Prepare sequences (12 months → predict next month).
   - Normalize data with **MinMaxScaler**.
   - Train LSTM model.
   - Evaluate with RMSE, MAE.
   - Found in **`notebooks/Model_LSTM_Retail.ipynb`**.

4. **API Deployment**:
   - Serve the trained model using **FastAPI**.
   - Found in **`app/main.py`**.

## 🚀 Running the FastAPI App

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the API

```bash
uvicorn app.main:app --reload
```

### API Endpoints

- **GET /**:
  - Test the API.
  - Response: `{ "message": "Demand Forecasting LSTM API is running!" }`

- **POST /predict/**:
  - Predict next month's demand based on last 12 months.
  - Request Body:

    ```json
    {
      "sequence": [2000, 2050, 3000, 2800, 2500, 2700, 2100, 2200, 2300, 2400, 2500, 2600]
    }
    ```

  - Response:

    ```json
    {
      "predicted_demand": 2600.75
    }
    ```

- **Interactive Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## ⚙️ How to Use This Project

1. Clone the repository.
2. Download the dataset from Kaggle and place in **`data/raw/`**.
3. Run EDA and Model notebooks in **`notebooks/`**.
4. Train and save your model in **`models/`**.
5. Run **FastAPI** app for real-time predictions.

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.x
- FastAPI
- Uvicorn
- Pandas, Numpy, Matplotlib, Scikit-learn, Joblib

## 💡 Future Enhancements

- Multi-item demand forecasting.
- Multi-variate LSTM (include prices, promotions).
- Batch prediction support.
- Cloud deployment (AWS, Azure, Heroku).

## 👤 Author

**Sujith Somanunnithan**

## 🔗 Related Resources

- [LSTM Time Series Guide (TensorFlow)](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
