import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from customer_segmentation import logger
from customer_segmentation.utils import CustomException
from customer_segmentation.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_model_bundle(self):
        try:
            logger.info(f"Loading model bundle from {self.config.model_path}")
            bundle = joblib.load(self.config.model_path)
            scaler = joblib.load(self.config.scaler_path)
            df = pd.read_csv(self.config.data_processed)
            X = df[bundle["feature_cols"]].fillna(df[bundle["feature_cols"]].median())
            X_scaled = scaler.transform(X)
            labels = bundle["model"].predict(X_scaled)
            df["Cluster"] = labels
            df["Segment"] = df["Cluster"].map(bundle["label_map"])
            logger.info(f"Data loaded: {df.shape[0]:,} rows, {X_scaled.shape[1]} features")
            return df, X_scaled, labels, bundle["feature_cols"]
        except Exception as e:
            raise CustomException(e, sys) from e

    def compute_metrics(self, X: np.ndarray, labels: np.ndarray) -> dict:
        try:
            sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
            dbi = davies_bouldin_score(X, labels)
            chi = calinski_harabasz_score(X, labels)
            metrics = {
                "n_clusters": int(len(np.unique(labels))),
                "n_samples": int(len(X)),
                "silhouette_score": round(float(sil), 6),
                "davies_bouldin_index": round(float(dbi), 6),
                "calinski_harabasz_score": round(float(chi), 2),
            }
            logger.info("Evaluation Metrics:")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v}")
            return metrics
        except Exception as e:
            raise CustomException(e, sys) from e

    def plot_silhouette(self, X: np.ndarray, labels: np.ndarray) -> str:
        try:
            sil_vals = silhouette_samples(X, labels)
            n_clusters = len(np.unique(labels))
            avg_score = sil_vals.mean()

            fig, ax = plt.subplots(figsize=(10, 6))
            y_lower = 10
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

            for cluster, color in zip(sorted(np.unique(labels)), colors):
                cluster_sil = np.sort(sil_vals[labels == cluster])
                y_upper = y_lower + len(cluster_sil)
                ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, alpha=0.7, color=color)
                ax.text(-0.05, y_lower + len(cluster_sil) / 2, str(cluster), fontsize=9)
                y_lower = y_upper + 10

            ax.axvline(x=avg_score, color="red", linestyle="--", label=f"Avg={avg_score:.3f}")
            ax.set_title(f"Silhouette Plot (K={n_clusters})", fontsize=13, fontweight="bold")
            ax.set_xlabel("Silhouette Coefficient")
            ax.set_ylabel("Cluster")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_path = str(self.config.artifacts_dir / "silhouette_plot.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Silhouette plot saved to {save_path}")
            return save_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def plot_cluster_distribution(self, df: pd.DataFrame) -> str:
        try:
            counts = df["Segment"].value_counts()
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(counts.index, counts.values, color=plt.cm.tab10.colors[:len(counts)])
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                        f"{val:,}\n({val / len(df) * 100:.1f}%)",
                        ha="center", va="bottom", fontsize=9)
            ax.set_title("Customer Segment Distribution", fontsize=13, fontweight="bold")
            ax.set_xlabel("Segment")
            ax.set_ylabel("Number of Customers")
            ax.grid(True, alpha=0.3, axis="y")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            save_path = str(self.config.artifacts_dir / "cluster_distribution.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Cluster distribution plot saved to {save_path}")
            return save_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def plot_feature_heatmap(self, df: pd.DataFrame, feature_cols: list) -> str:
        try:
            available = [f for f in feature_cols[:15] if f in df.columns]
            cluster_means = df.groupby("Cluster")[available].mean()
            cluster_means_norm = (cluster_means - cluster_means.min()) / (
                cluster_means.max() - cluster_means.min() + 1e-9
            )
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.heatmap(
                cluster_means_norm.T,
                annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Normalised Mean"},
            )
            ax.set_title("Cluster Feature Heatmap (Normalised)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Feature")
            plt.tight_layout()
            save_path = str(self.config.artifacts_dir / "feature_heatmap.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Feature heatmap saved to {save_path}")
            return save_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def run(self):
        try:
            load_dotenv()
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            mlflow.set_experiment(self.config.experiment_name)

            df, X_scaled, labels, feature_cols = self.load_model_bundle()
            metrics = self.compute_metrics(X_scaled, labels)

            with open(self.config.metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to {self.config.metrics_path}")

            sil_path = self.plot_silhouette(X_scaled, labels)
            dist_path = self.plot_cluster_distribution(df)
            heatmap_path = self.plot_feature_heatmap(df, feature_cols)

            with mlflow.start_run(run_name="evaluation"):
                mlflow.log_metrics({
                    "eval_silhouette_score": metrics["silhouette_score"],
                    "eval_davies_bouldin_index": metrics["davies_bouldin_index"],
                    "eval_calinski_harabasz_score": metrics["calinski_harabasz_score"],
                })
                mlflow.log_param("n_clusters", metrics["n_clusters"])
                mlflow.log_artifact(str(self.config.metrics_path))
                mlflow.log_artifact(sil_path)
                mlflow.log_artifact(dist_path)
                mlflow.log_artifact(heatmap_path)
                logger.info("Evaluation run logged to MLflow/DagsHub")

            return metrics
        except Exception as e:
            raise CustomException(e, sys) from e
