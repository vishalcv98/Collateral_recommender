import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

class RefundRiskModel:
    def __init__(self, trans_df, accounts_df, cutoff_date="2024-01-01", reference_date="2024-12-01"):
        self.trans_df = trans_df
        self.accounts_df = accounts_df
        self.cutoff_date = pd.to_datetime(cutoff_date)
        self.reference_date = pd.to_datetime(reference_date)

    def preprocess(self):
        print("Beginning Preprocessing")
        self.trans_df['TRANSACTION_ON'] = pd.to_datetime(self.trans_df['TRANSACTION_ON'], errors='coerce')
        self.trans_df.dropna(subset=['TRANSACTION_ON'], inplace=True)
        self.past_txns = self.trans_df[self.trans_df['TRANSACTION_ON'] < self.cutoff_date]
        self.future_txns = self.trans_df[self.trans_df['TRANSACTION_ON'] >= self.cutoff_date]
        print("Preprocessing complete")

    def create_labels(self):
        print("Creating labels")
        future_refunds = self.future_txns.groupby("COMPANY_ID")['AMOUNT_REFUNDED'].sum().reset_index()
        future_refunds['refund_flag'] = (future_refunds['AMOUNT_REFUNDED'] > 0).astype(int)
        self.future_refunds = future_refunds[['COMPANY_ID', 'refund_flag']]
        print("Labels created")

    def generate_features(self):
        print("BGenerating Features")
        features = self.past_txns.groupby("COMPANY_ID").agg(
            num_transactions=("FF_ID", "count"),
            total_payment_usd=("PAYMENT_AMOUNT", "sum"),
            total_refunded_usd=("AMOUNT_REFUNDED", "sum"),
            avg_payment=("PAYMENT_AMOUNT", "mean"),
            max_payment=("PAYMENT_AMOUNT", "max"),
            num_refunds=("AMOUNT_REFUNDED", lambda x: (x > 0).sum()),
            num_disputes=("IS_DISPUTE", "sum")
        ).reset_index()

        features['refund_rate'] = features['total_refunded_usd'] / features['total_payment_usd'].replace(0, 1)

        txn_dates = self.past_txns.groupby("COMPANY_ID").agg(
            last_txn_date=("TRANSACTION_ON", "max"),
            first_txn_date=("TRANSACTION_ON", "min")
        ).reset_index()
        txn_dates['days_since_last_txn'] = (self.reference_date - txn_dates['last_txn_date']).dt.days
        txn_dates['active_days'] = (txn_dates['last_txn_date'] - txn_dates['first_txn_date']).dt.days

        txn_counts = self.past_txns.groupby("COMPANY_ID").agg(
            total_txns=("FF_ID", "count"),
            txn_period_days=("TRANSACTION_ON", lambda x: (x.max() - x.min()).days + 1)
        ).reset_index()
        txn_counts['txns_per_month'] = txn_counts['total_txns'] / (txn_counts['txn_period_days'] / 30.0)

        last_90d_cutoff = self.reference_date - pd.Timedelta(days=90)
        recent = self.past_txns[self.past_txns['TRANSACTION_ON'] >= last_90d_cutoff]
        recent_features = recent.groupby("COMPANY_ID").agg(
            recent_txns=("FF_ID", "count"),
            recent_total_paid=("PAYMENT_AMOUNT", "sum"),
            recent_refunds=("AMOUNT_REFUNDED", "sum")
        ).reset_index()
        recent_features['recent_refund_ratio'] = recent_features['recent_refunds'] / recent_features['recent_total_paid'].replace(0, 1)

        features = features.merge(txn_dates, on='COMPANY_ID', how='left')
        features = features.merge(txn_counts, on='COMPANY_ID', how='left')
        features = features.merge(recent_features, on='COMPANY_ID', how='left')

        vendor_info = self.accounts_df[[
            "COMPANY_ID", "COMPANY_TYPE", "MEMBER_SEGMENT_ATV", "MONTHS_AGE",
            "NO_OF_EMPLOYEES", "GPV_2YEARS", "PROJECTS_2YEARS", "PAYMENTS_NOT_MANUAL"
        ]]
        vendor_info['COMPANY_TYPE'] = vendor_info['COMPANY_TYPE'].astype(str)
        vendor_info['MEMBER_SEGMENT_ATV'] = vendor_info['MEMBER_SEGMENT_ATV'].astype(str)
        vendor_info = pd.get_dummies(vendor_info, columns=["COMPANY_TYPE", "MEMBER_SEGMENT_ATV"], dummy_na=True)
        dummy_cols = vendor_info.columns.difference(["COMPANY_ID", "MONTHS_AGE", "NO_OF_EMPLOYEES",
                                                     "PAYMENTS_NOT_MANUAL", "GPV_2YEARS", "PROJECTS_2YEARS"])
        vendor_info[dummy_cols] = vendor_info[dummy_cols].astype(int)

        features = features.merge(vendor_info, on='COMPANY_ID', how='left')
        features = features.merge(self.future_refunds, on='COMPANY_ID', how='left')
        features.dropna(subset=['refund_flag'], inplace=True)
        features.drop(['NO_OF_EMPLOYEES','last_txn_date','first_txn_date'], axis=1, inplace=True, errors='ignore')
        features = features.loc[:, ~features.columns.duplicated()]
        self.df = features
        self.feature_columns = [col for col in features.columns if col not in ['refund_flag']]
        joblib.dump(self.feature_columns, 'feature_matcher\\feature_columns.joblib')
        print("Features generated")

    def train_model(self):
        print("Training model")
        X = self.df.drop(columns=["refund_flag"])
        y = self.df["refund_flag"]

        X_train, X_test, y_train, y_test = train_test_split(
            X.drop(columns=["COMPANY_ID"]), y, test_size=0.2, stratify=y, random_state=42
        )

        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos

        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=1,
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight * 0.8,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")

        self.df['predicted_refund_risk'] = model.predict_proba(X.drop(columns=['COMPANY_ID']))[:, 1]
        self.model = model
        joblib.dump(model, 'model_file\\risky_vendir_classifier.joblib')
        print("Model trained")

# Replace with path of Transaction Data
chunks = pd.read_csv('Copy of crediarc_transactions_full.csv',
                       chunksize=500000)
trans_df = pd.concat(chunks, ignore_index=True)

# Replace with path of Accounts Data
accounts_df = pd.read_csv('crediarc_accounts_full_nn.csv')
print('data successfully read')
model = RefundRiskModel(trans_df, accounts_df)
model.preprocess()
model.create_labels()
model.generate_features()
model.train_model()