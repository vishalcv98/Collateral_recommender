import pandas as pd
import numpy as np
from datetime import timedelta
import joblib
from xgboost import XGBClassifier

class RefundRiskPredictor:
    def __init__(self, model):
        self.model = model
        self.min_prob_threshold = 0.4
        self.weight_pred_min = 0.3
        self.weight_pred_max = 0.9

    def preprocess(self, trans_df, accounts_df):
        # Ensure datetime
        trans_df = trans_df.copy()
        trans_df['TRANSACTION_ON'] = pd.to_datetime(trans_df['TRANSACTION_ON'])

        # Cutoff for feature engineering
        cutoff = trans_df['TRANSACTION_ON'].max()

        past_txns = trans_df[trans_df["TRANSACTION_ON"] < cutoff]

        # ====== Historical Features ======
        features = past_txns.groupby("COMPANY_ID").agg(
            num_transactions=("FF_ID", "count"),
            total_payment_usd=("PAYMENT_AMOUNT", "sum"),
            total_refunded_usd=("AMOUNT_REFUNDED", "sum"),
            avg_payment=("PAYMENT_AMOUNT", "mean"),
            max_payment=("PAYMENT_AMOUNT", "max"),
            num_refunds=("AMOUNT_REFUNDED", lambda x: (x > 0).sum()),
            num_disputes=("IS_DISPUTE", "sum"),
        ).reset_index()

        features["refund_rate"] = features["total_refunded_usd"] / features["total_payment_usd"].replace(0, 1)

        # ====== Recency Features ======
        recency = past_txns.groupby("COMPANY_ID").agg(
            last_txn_date=("TRANSACTION_ON", "max"),
            first_txn_date=("TRANSACTION_ON", "min")
        ).reset_index()

        recency["days_since_last_txn"] = (cutoff - recency["last_txn_date"]).dt.days
        recency["active_days"] = (recency["last_txn_date"] - recency["first_txn_date"]).dt.days
        features = features.merge(recency[["COMPANY_ID", "days_since_last_txn", "active_days"]], on="COMPANY_ID", how="left")

        # ====== Frequency Features ======
        txn_counts = past_txns.groupby("COMPANY_ID").agg(
            total_txns=("FF_ID", "count"),
            txn_period_days=("TRANSACTION_ON", lambda x: (x.max() - x.min()).days + 1)
        ).reset_index()
        txn_counts["txns_per_month"] = txn_counts["total_txns"] / (txn_counts["txn_period_days"] / 30.0)
        features = features.merge(txn_counts, on="COMPANY_ID", how="left")

        # ====== Recent 90-day Features ======
        last_90d_cutoff = cutoff - timedelta(days=90)
        recent = past_txns[past_txns["TRANSACTION_ON"] >= last_90d_cutoff]
        recent_agg = recent.groupby("COMPANY_ID").agg(
            recent_txns=("FF_ID", "count"),
            recent_total_paid=("PAYMENT_AMOUNT", "sum"),
            recent_refunds=("AMOUNT_REFUNDED", "sum")
        ).reset_index()
        recent_agg["recent_refund_ratio"] = recent_agg["recent_refunds"] / recent_agg["recent_total_paid"].replace(0, 1)
        features = features.merge(recent_agg, on="COMPANY_ID", how="left")

        # ====== Account Features ======
        accounts_df = accounts_df.copy()
        accounts_df['COMPANY_TYPE'] = accounts_df['COMPANY_TYPE'].astype(str)
        accounts_df['MEMBER_SEGMENT_ATV'] = accounts_df['MEMBER_SEGMENT_ATV'].astype(str)
        accounts_df = pd.get_dummies(accounts_df, columns=["COMPANY_TYPE", "MEMBER_SEGMENT_ATV"], dummy_na=True)
        print(accounts_df.columns)
        dummy_cols = accounts_df.columns.difference(['ACCOUNT_ID', 'MONTHS_AGE', 'COMPANY_ID', 
                                                     'ADDRESS', 'WEBSITE_URL',
       'NO_OF_EMPLOYEES', 'PAYMENTS_NOT_MANUAL', 'GPV_2YEARS',
       'PROJECTS_2YEARS',])
        accounts_df[dummy_cols] = accounts_df[dummy_cols].astype(int)
        features = features.merge(accounts_df, on="COMPANY_ID", how="left")
        features = features.drop(columns=["NO_OF_EMPLOYEES",'last_txn_date','first_txn_date'], errors='ignore')
        expected_cols = joblib.load('feature_columns.joblib')
        for col in expected_cols:
            if col not in features.columns:
                features[col] = 0
        features = features[expected_cols]
        return features
    
    def predict(self, trans_df, accounts_df):
        df = self.preprocess(trans_df, accounts_df)

        company_ids = df["COMPANY_ID"]
        X = df.drop(columns=["COMPANY_ID"])
        X = X.loc[:, ~X.columns.duplicated()]
        df["predicted_refund_risk"] = self.model.predict_proba(X)[:, 1]
        return df