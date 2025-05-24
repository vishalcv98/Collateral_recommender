import pandas as pd
import numpy as np
from scripts.predict_risk import RefundRiskPredictor
import joblib

# Replace with path of Transaction Data
chunks = pd.read_csv('Copy of crediarc_transactions_full.csv',
                       chunksize=500000)
trans_df = pd.concat(chunks, ignore_index=True)

# Replace with path of Accounts Data
accounts_df = pd.read_csv('crediarc_accounts_full_nn.csv')
print('data successfully read')
model = joblib.load("model_file\\risky_vendir_classifier.joblib")
predictor = RefundRiskPredictor(model)

# Load your fresh transaction and account data
df = predictor.predict(trans_df, accounts_df)
df = df.loc[:, ~df.columns.duplicated()]
print(df.columns)
print(df.head())


def recommend_collateral_dynamic_weight_extended(row, 
                                                weight_pred_min=0.3, 
                                                weight_pred_max=0.9, 
                                                min_prob_threshold=0.4):
    pred_prob = row["predicted_refund_risk"]
    hist_rate = row["refund_rate"]
    
    # Handle vendors with few transactions (fallback)
    if row["num_transactions"] < 5:
        # Find MEMBER_SEGMENT_ATV columns
        atv_cols = [c for c in df.columns if c.startswith("MEMBER_SEGMENT_ATV_")]
        atv_col = next((c for c in atv_cols if row[c] == 1), None)
        
        # Find COMPANY_TYPE columns
        comp_cols = [c for c in df.columns if c.startswith("COMPANY_TYPE_")]
        comp_col = next((c for c in comp_cols if row[c] == 1), None)
        
        # Try MEMBER_SEGMENT_ATV first
        if comp_col:
            hist_rate = df[df[comp_col] == 1]["refund_rate"].mean()
        # Then try COMPANY_TYPE
        elif atv_col:
            hist_rate = df[df[atv_col] == 1]["refund_rate"].mean()
        # Else fallback to global mean
        else:
            hist_rate = df["refund_rate"].mean()
    
    if pred_prob < min_prob_threshold:
        return (round(min_prob_threshold*0.2 / 0.05) * 0.05)
    
    scale = (pred_prob - min_prob_threshold) / (1 - min_prob_threshold)
    weight_pred = weight_pred_min + scale * (weight_pred_max - weight_pred_min)
    weight_hist = 1 - weight_pred
    
    est_risk = weight_pred * pred_prob + weight_hist * hist_rate
    
    collateral_pct = np.clip(est_risk, 0.05, 0.50)
    collateral_pct_rounded = round(collateral_pct / 0.05) * 0.05
    
    return collateral_pct_rounded
print(df.columns)
df["recommended_collateral_pct"] = df.apply(recommend_collateral_dynamic_weight_extended, axis=1)
print(df.head())