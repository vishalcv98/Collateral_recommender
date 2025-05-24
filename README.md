# Methodology
Overview
This project aims to determine an appropriate collateral percentage for vendors based on their predicted refund risk and historical refund behavior. The system blends model-driven risk estimation with historical heuristics, adjusting the weight of each source dynamically based on prediction confidence.

# Input Features
The dataset consists of over 70 features per vendor, including:

- Transaction aggregates: total payments, number of transactions, refund counts, average/max payment, etc.

- Temporal features: days since last transaction, active days, recent transaction/refund patterns.

- Business metadata: one-hot encoded COMPANY_TYPE and MEMBER_SEGMENT_ATV attributes.

- Model output: predicted_refund_risk, a probability score from a trained classifier.

# Collateral Recommendation Strategy
The function recommend_collateral_dynamic_weight_extended() estimates the recommended collateral percentage for each vendor. This is calculated as a weighted blend of:

Model-predicted refund risk (confidence-based)

Historical refund rate (segment or global average fallback)

Here’s a breakdown of how it works:

Step-by-Step Breakdown
- Handle Low Transaction Volume
If the vendor has fewer than 5 transactions, their personal refund_rate is likely unreliable. In such cases:

Attempt to use segment-level refund rates:

First by matching the COMPANY_TYPE_* column where value is 1.

If not found, try matching the MEMBER_SEGMENT_ATV_* column.

If both fail, fall back to the global average refund rate from all vendors.

This ensures that vendors with minimal transaction history still receive a reasonable estimate grounded in peer behavior.

- Confidence-Based Blending
Once we have:

pred_prob: model’s predicted refund probability for the vendor

hist_rate: historical refund rate (actual or fallback)

We use dynamic blending to compute an overall estimated risk:

## scale = (pred_prob - min_prob_threshold) / (1 - min_prob_threshold)
## weight_pred = weight_pred_min + scale * (weight_pred_max - weight_pred_min)
## weight_hist = 1 - weight_pred
weight_pred_min and weight_pred_max define the minimum and maximum trust in the model's output.

min_prob_threshold defines the confidence floor. Predictions below this threshold are treated as noise.

# Key Insight:

If pred_prob is close to the threshold → rely more on hist_rate.

If pred_prob is much higher than the threshold → rely more on pred_prob.

This leads to the final blended risk:

## est_risk = weight_pred * pred_prob + weight_hist * hist_rate
- Clip and Round Collateral
To make the recommendation business-friendly:

## We clip the est_risk to stay between 5% and 50%:

collateral_pct = np.clip(est_risk, 0.05, 0.50)
Then, we round it to the nearest 5% increment for ease of interpretation:

collateral_pct_rounded = round(collateral_pct / 0.05) * 0.05
This becomes the final recommended collateral percentage for the vendor.

Final Application
The function is vectorized over the DataFrame:

df["recommended_collateral_pct"] = df.apply(recommend_collateral_dynamic_weight_extended, axis=1)
# Why This Approach?
Robustness: Falls back to reliable averages for vendors with sparse data.

Adaptability: Gives higher weight to predictions when model confidence is high.

Interpretability: Rounds and caps results for easier operational use.

Fairness: Avoids penalizing new vendors with limited data by relying on segment-level risk norms.

# Risk Predictor model results on test set

               precision    recall  f1-score   support

         0.0       0.87      0.81      0.84      9428
         1.0       0.56      0.65      0.60      3394

    accuracy                           0.77     12822
   macro avg       0.71      0.73      0.72     12822
weighted avg       0.78      0.77      0.78     12822

ROC AUC Score: 0.8168

Threshold Precision Recall    F1-score  
0.1       0.292     0.984     0.450     
0.2       0.355     0.932     0.514     
0.3       0.426     0.855     0.568     
0.4       0.487     0.763     0.595     
0.5       0.556     0.651     0.600     
0.6       0.632     0.527     0.575     
0.7       0.734     0.388     0.508     
0.8       0.836     0.253     0.388     
0.9       0.937     0.159     0.271  


# How to Run the Code
## Prerequisites
Python 3.8+

Required libraries: pandas, numpy, scikit-learn, etc.

Activate your Python virtual environment and install dependencies from requirements.txt if available.

# Execution Steps
- `git clone https://github.com/vishalcv98/Collateral_recommender.git`
- `cd Collateral_recommender`
- Make sure you have Python 3.13.3 installed on your system. To create a virtual environment using venv: 


- run `python3.13 -m venv venv`

- run `source venv/bin/activate`

- run `pip install -r requirements.txt` in the terminal

- Update file paths

Open the following two files:

scripts/training.py

scripts/recommend_collateral.py

➤ Locate the section with comments indicating where to provide paths for the input transactions and accounts data files, and update accordingly.

Run Training Script (only once unless retraining)

This will generate the model for refund risk prediction.

python scripts/training.py
Run Recommendation Script

This applies the trained model and generates collateral recommendations.

python scripts/recommend_collateral.py

- Check Output

Results will be stored in the results/ folder as a .csv file with recommended collateral percentages.
