# Customer Segmentation with RFM & K-Means Clustering

## 1. Project Overview 🎯

This project analyzes a transactional dataset from an online retailer to perform customer segmentation. The goal is to identify distinct customer groups based on their purchasing behavior using the **RFM (Recency, Frequency, Monetary)** model and the **K-Means clustering** algorithm. By understanding these segments, we can provide actionable insights for targeted marketing strategies.

## 2. Dataset 📦

The data is the "Online Retail II" dataset, which contains over 1 million transactions from a UK-based online retailer.

* **Source:** UCI Machine Learning Repository / Kaggle
* **Time Period:** 2010-2011
* **Key Attributes:** `InvoiceNo`, `StockCode`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

## 3. Methodology / Workflow ⚙️

The project was executed in a Jupyter Notebook (`.ipynb`) and followed these key steps:

1.  **Data Cleaning & Preprocessing:**
    * Loaded 1M+ raw transaction records.
    * Handled missing `CustomerID` values (dropped).
    * Removed negative `Quantity` (returns) and zero-price items.
    * Converted `InvoiceDate` to datetime objects.

2.  **Feature Engineering (RFM):**
    * Calculated three key metrics for each of the 5,800+ unique customers:
        * **Recency (R):** Days since the last purchase.
        * **Frequency (F):** Total number of unique invoices.
        * **Monetary (M):** Total monetary value of purchases.

3.  **Data Scaling & Transformation:**
    * Applied **Log Transformation** (`np.log1p`) to normalize the heavily right-skewed RFM data.
    * Used **StandardScaler** (Z-score) to scale all features to a common range, making them suitable for K-Means.

4.  **Modeling (K-Means Clustering):**
    * Used the **Elbow Method** to determine the optimal number of clusters (K).
    * Selected **K=4** as the optimal value.
    * Trained the K-Means model on the scaled data and assigned each customer to one of the 4 clusters.

## 4. Key Findings & Analysis 📈

The model successfully identified 4 distinct customer personas based on their RFM scores:

* **Cluster 0: 🏆 VIP / Loyal Customers**
    * *(R: Low, F: High, M: High)* - High-value, frequent buyers who purchased recently.
    * **Action:** Reward with loyalty programs, exclusive access, and seek feedback.

* **Cluster 3: ✨ New / Potential Customers**
    * *(R: Low, F: Low, M: Low)* - New customers who purchased recently but with low frequency.
    * **Action:** Nurture with welcome emails, onboarding, and incentives for a second purchase.

* **Cluster 2: ⚠️ At-Risk Customers**
    * *(R: High, F: Medium, M: Medium)* - Good customers who *used* to buy, but haven't returned in a long time.
    * **Action:** Target with "We miss you!" win-back campaigns and strong discounts.

* **Cluster 1: 💤 Lost / Inactive Customers**
    * *(R: Very High, F: Low, M: Low)* - Customers who purchased once long ago and never returned.
    * **Action:** Low-cost re-engagement (e.g., general newsletter). Do not spend a high marketing budget here.



## 5. Technologies Used 💻

* **Python**
* **Pandas** (Data manipulation and cleaning)
* **NumPy** (Numerical operations)
* **Scikit-learn (sklearn)** (StandardScaler, KMeans)
* **Matplotlib & Seaborn** (Data visualization)
* **Jupyter Notebook** (Project environment)
