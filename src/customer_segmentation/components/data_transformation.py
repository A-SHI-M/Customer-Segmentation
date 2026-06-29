import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from customer_segmentation import logger
from customer_segmentation.utils import CustomException
from customer_segmentation.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        np.random.seed(config.random_seed)

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Validating data ...")
            original_len = len(df)
            df = df.drop_duplicates()
            logger.info(f"Dropped {original_len - len(df):,} duplicate rows")

            if "Age" in df.columns:
                df = df[df["Age"].between(5, 100) | df["Age"].isna()]

            if "Total_Purchases" in df.columns:
                df.loc[df["Total_Purchases"] < 0, "Total_Purchases"] = np.nan

            logger.info(f"Validated dataset: {len(df):,} rows remain")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Preprocessing data ...")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in num_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)

            if "Gender" in df.columns:
                df["Gender_Encoded"] = (df["Gender"] == "Male").astype(int)

            if "Signup_Quarter" in df.columns:
                quarter_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
                df["Signup_Quarter_Encoded"] = df["Signup_Quarter"].map(quarter_map).fillna(2)

            logger.info("Preprocessing complete")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Engineering features ...")
            df["Total_Spending"] = df["Lifetime_Value"]
            df["Purchase_Frequency"] = df["Total_Purchases"] / df["Membership_Years"].clip(lower=0.1)
            df["RFM_Recency"] = 1 / (df["Days_Since_Last_Purchase"] + 1)
            df["RFM_Frequency"] = df["Total_Purchases"]
            df["RFM_Monetary"] = df["Lifetime_Value"]
            df["Customer_Age_Group"] = pd.cut(
                df["Age"],
                bins=[0, 30, 50, 200],
                labels=[0, 1, 2],
            ).astype(float)
            logger.info("Feature engineering complete")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def scale_features(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        try:
            logger.info(f"Scaling {len(feature_cols)} features ...")
            X = df[feature_cols].copy()
            X = X.fillna(X.median())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            scaler_path = self.config.models_dir / "scaler.pkl"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
            return X_scaled
        except Exception as e:
            raise CustomException(e, sys) from e

    def run(self):
        try:
            logger.info(f"Loading data from {self.config.data_raw}")
            df = pd.read_csv(self.config.data_raw)
            logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

            df = self.validate_data(df)
            df = self.preprocess_data(df)
            df = self.engineer_features(df)

            all_features = [
                f for f in self.config.numerical_features + self.config.engineered_features
                if f in df.columns
            ]

            X_scaled = self.scale_features(df, all_features)

            df.to_csv(self.config.data_processed, index=False)
            logger.info(f"Processed data saved to {self.config.data_processed}")
            logger.info(f"Feature matrix shape: {X_scaled.shape}")

            return df, X_scaled, all_features
        except Exception as e:
            raise CustomException(e, sys) from e
