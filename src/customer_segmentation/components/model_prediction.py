import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

from customer_segmentation import logger
from customer_segmentation.utils import CustomException
from customer_segmentation.entity.config_entity import ModelPredictionConfig

SEGMENT_DESCRIPTIONS = {
    "VIP Customers": {
        "description": "Highest lifetime value; premium spenders with frequent, recent purchases.",
        "strategy": "Exclusive loyalty rewards, early-access offers, dedicated account manager.",
        "icon": "👑",
    },
    "Loyal Customers": {
        "description": "Regular buyers with strong engagement and consistent spending.",
        "strategy": "Loyalty programs, personalised recommendations, birthday discounts.",
        "icon": "⭐",
    },
    "Potential Customers": {
        "description": "Moderate spenders showing growth signals; not yet fully committed.",
        "strategy": "Targeted upsell campaigns, product education, limited-time offers.",
        "icon": "🚀",
    },
    "New Customers": {
        "description": "Recently acquired; low tenure and purchase history.",
        "strategy": "Welcome series, onboarding guides, first-purchase incentives.",
        "icon": "🌱",
    },
    "At-Risk Customers": {
        "description": "Low recent activity; at risk of churning.",
        "strategy": "Win-back campaigns, personalised outreach, reactivation discounts.",
        "icon": "⚠️",
    },
}


class ModelPrediction:
    def __init__(self, config: ModelPredictionConfig):
        self.config = config

    def load_artifacts(self):
        try:
            bundle = joblib.load(self.config.model_path)
            scaler = joblib.load(self.config.scaler_path)
            logger.info(f"Model bundle loaded from {self.config.model_path}")
            return bundle["model"], scaler, bundle["label_map"], bundle["feature_cols"]
        except Exception as e:
            raise CustomException(e, sys) from e

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Total_Spending"] = df.get("Lifetime_Value", df.get("Total_Spending", 0))
        membership = df["Membership_Years"] if "Membership_Years" in df.columns else pd.Series(np.ones(len(df)))
        df["Purchase_Frequency"] = df.get("Total_Purchases", pd.Series(np.zeros(len(df)))) / membership.clip(lower=0.1)
        days = df["Days_Since_Last_Purchase"] if "Days_Since_Last_Purchase" in df.columns else pd.Series([30] * len(df))
        df["RFM_Recency"] = 1 / (days + 1)
        df["RFM_Frequency"] = df.get("Total_Purchases", pd.Series(np.zeros(len(df))))
        df["RFM_Monetary"] = df.get("Lifetime_Value", pd.Series(np.zeros(len(df))))
        age = df["Age"] if "Age" in df.columns else pd.Series([35] * len(df))
        df["Customer_Age_Group"] = pd.cut(age, bins=[0, 30, 50, 200], labels=[0, 1, 2]).astype(float)
        return df

    def _preprocess_input(self, df: pd.DataFrame, feature_cols: list, scaler) -> np.ndarray:
        if "Gender" in df.columns:
            df["Gender_Encoded"] = (df["Gender"] == "Male").astype(int)
        df = self._engineer_features(df)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df[feature_cols] = df[feature_cols].fillna(0)
        return scaler.transform(df[feature_cols].values)

    def predict(self, input_data: Union[pd.DataFrame, dict, str]) -> pd.DataFrame:
        try:
            model, scaler, label_map, feature_cols = self.load_artifacts()

            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, str):
                df = pd.read_csv(input_data)
            else:
                df = input_data.copy()

            logger.info(f"Predicting for {len(df):,} customer(s) ...")
            X_scaled = self._preprocess_input(df, feature_cols, scaler)
            clusters = model.predict(X_scaled)

            df = df.copy()
            df["Cluster"] = clusters
            df["Segment"] = df["Cluster"].map(label_map)
            df["Segment_Description"] = df["Segment"].map(
                lambda s: SEGMENT_DESCRIPTIONS.get(s, {}).get("description", "Unknown")
            )
            df["Marketing_Strategy"] = df["Segment"].map(
                lambda s: SEGMENT_DESCRIPTIONS.get(s, {}).get("strategy", "Unknown")
            )
            df["Icon"] = df["Segment"].map(
                lambda s: SEGMENT_DESCRIPTIONS.get(s, {}).get("icon", "❓")
            )

            logger.info("Segment distribution:")
            for seg, cnt in df["Segment"].value_counts().items():
                logger.info(f"  {seg}: {cnt:,} customers")

            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def run(self):
        try:
            sample_customer = {
                "Age": 35,
                "Gender": "Female",
                "Membership_Years": 3.5,
                "Login_Frequency": 15,
                "Session_Duration_Avg": 30,
                "Pages_Per_Session": 9,
                "Cart_Abandonment_Rate": 50,
                "Wishlist_Items": 5,
                "Total_Purchases": 20,
                "Average_Order_Value": 120,
                "Days_Since_Last_Purchase": 10,
                "Discount_Usage_Rate": 40,
                "Returns_Rate": 5,
                "Email_Open_Rate": 25,
                "Customer_Service_Calls": 4,
                "Product_Reviews_Written": 3,
                "Social_Media_Engagement_Score": 30,
                "Mobile_App_Usage": 20,
                "Payment_Method_Diversity": 2,
                "Lifetime_Value": 1500,
                "Credit_Balance": 2000,
                "Signup_Quarter": "Q2",
                "Country": "USA",
                "City": "New York",
            }

            result = self.predict(sample_customer)
            result.to_csv(self.config.predictions_path, index=False)
            logger.info(f"Prediction saved to {self.config.predictions_path}")

            seg = result["Segment"].iloc[0]
            icon = result["Icon"].iloc[0]
            desc = result["Segment_Description"].iloc[0]
            strat = result["Marketing_Strategy"].iloc[0]
            cluster = result["Cluster"].iloc[0]

            logger.info("=" * 60)
            logger.info("PREDICTION RESULT")
            logger.info("=" * 60)
            logger.info(f"Cluster    : {cluster}")
            logger.info(f"Segment    : {icon} {seg}")
            logger.info(f"Description: {desc}")
            logger.info(f"Strategy   : {strat}")
            logger.info("=" * 60)

            return result
        except Exception as e:
            raise CustomException(e, sys) from e
