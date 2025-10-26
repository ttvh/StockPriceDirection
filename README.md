# Short-Term Stock Price Direction Prediction (MSN Ticker)

This project builds a complete machine learning pipeline to predict the short-term price direction (Up/Down) of the MSN stock ticker. The goal is to classify whether the price will rise or fall in the next 2-minute interval.

This notebook demonstrates a full data science workflow:
1.  **Data Ingestion & Cleaning:** Loading and preparing raw 15-minute data.
2.  **Resampling & Feature Engineering:** Converting the data to a 2-minute frequency and creating over 20 technical indicators.
3.  **Modeling:** Training and comparing three different classification models (Logistic Regression, Random Forest, LSTM).
4.  **Handling Imbalance:** Solving a critical class imbalance problem to create a useful model.
5.  **Evaluation:** Analyzing model performance not just on **Accuracy**, but on **Precision** and **Recall** to determine real-world effectiveness.

---

## 1. Data Pipeline

### a. Data Loading
The project starts with a raw 15-minute OHLCV (Open, High, Low, Close, Volume) dataset for the MSN ticker.

* **Raw Data Profile:** The initial dataset contains **135,354** entries. The `Date/Time` column is an `object` type and requires conversion.

### b. Resampling & Cleaning
To generate a larger dataset suitable for high-frequency prediction, the data is processed as follows:
1.  **Resample:** The 15-minute data is resampled to a 2-minute frequency (`resample('2min')`).
2.  **Clean:** All `NaN` rows (from non-trading times) and all intervals with `Volume = 0` are dropped, resulting in a clean dataset of active trading periods.

### c. Feature Engineering
A set of ~20 features was engineered to provide market context for the models:

* **Target Variable (`Target_Direction`):** The core prediction goal.
    * **1 (Up):** If the `Close` price 2 minutes from now is > current `Close` price.
    * **0 (Down):** If the `Close` price 2 minutes from now is $\le$ current `Close` price.
* **Price Lags/Momentum:** `Close_Lag` and `Pct_Change` for 1, 2, 3, 5, and 10 previous steps (2-20 minutes).
* **Technical Indicators (using `ta` library):**
    * **Trend:** `SMA` (10, 50), `MACD_diff`
    * **Momentum:** `RSI` (14)
    * **Volatility:** `BollingerBands (%B)`, `ATR` (14)
* **Volume Features:** `Volume_SMA_20`, `Volume_Pct_Change`

### d. Model Preparation
1.  **Train-Test Split:** Data is split 80% (Train) / 20% (Test) using `shuffle=False` to respect the time-series nature of the data.
2.  **Scaling:** All features are scaled using `MinMaxScaler`.
3.  **Handling Class Imbalance:** The target variable was highly imbalanced (e.g., ~72% "Down" vs. 28% "Up"). This was solved by applying `class_weight='balanced'` to all models, forcing them to pay more attention to the minority "Up" class.

---

## 2. Model Training & Results

Three distinct models were trained and evaluated on the unseen test set.

### a. Logistic Regression (Baseline)
* **Accuracy:** 61.17%
* **Precision (Up):** 0.40
* **Recall (Up):** 0.79
* **Analysis:** This simple model served as a baseline. While its overall accuracy was modest, it was surprisingly effective at identifying "Up" movements, correctly catching 79% of all "Up" signals.

### b. Random Forest (Tuned)
* **Tuning:** `GridSearchCV` was used to find the best hyperparameters.
* **Best Params:** `{'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 100}`
* **Accuracy:** 71.74%
* **Precision (Up):** 0.47
* **Recall (Up):** 0.04
* **Analysis:** This model achieved the **highest accuracy**, but this result is **highly misleading**. A Recall of only 4% for the "Up" class means the model almost *never* predicted a price increase. Its high accuracy came from simply guessing the majority "Down" class, making it useless for practical trading.

### c. LSTM (Deep Learning)
* **Accuracy:** 58.52%
* **Precision (Up):** 0.33
* **Recall (Up):** 0.47
* **Analysis:** The LSTM had the lowest accuracy but was the **most balanced model**. It demonstrated a genuine (though limited) ability to identify both "Down" (63% Recall) and "Up" (47% Recall) movements, unlike the Random Forest which "cheated" by ignoring the minority class.

---

## 3. Conclusion

1.  **Accuracy is Misleading:** This project clearly shows that `Accuracy` is a poor metric for imbalanced financial data. The 71.74% accurate Random Forest was the worst-performing model in practice.
2.  **The "Noise" Problem:** Predicting 2-minute price movements is an extremely "noisy" task. The models struggled to find strong, consistent signals.
3.  **Balanced Models:** The `Logistic Regression` and `LSTM` models, which were forced to handle class imbalance, provided more realistic and balanced (though less accurate) results. The `Logistic Regression` model showed the most promise for identifying "Up" signals.

## 4. Technologies Used
* Python
* Pandas & NumPy
* Scikit-learn (LogisticRegression, RandomForestClassifier, GridSearchCV, MinMaxScaler, metrics)
* TensorFlow (Keras) (Sequential, LSTM, Dropout)
* `ta` (Technical Analysis library)
* Matplotlib & Seaborn
