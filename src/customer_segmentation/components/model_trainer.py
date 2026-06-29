import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from customer_segmentation import logger
from customer_segmentation.utils import CustomException
from customer_segmentation.entity.config_entity import ModelTrainerConfig

SEGMENT_NAMES = [
    "VIP Customers",
    "Loyal Customers",
    "Potential Customers",
    "New Customers",
    "At-Risk Customers",
]


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_data(self):
        try:
            logger.info(f"Loading processed data from {self.config.data_processed}")
            df = pd.read_csv(self.config.data_processed)
            scaler = joblib.load(self.config.scaler_path)
            X = df[self.config.feature_cols].fillna(df[self.config.feature_cols].median())
            X_scaled = scaler.transform(X)
            logger.info(f"Data loaded: {df.shape[0]:,} rows, {X_scaled.shape[1]} features")
            return df, X_scaled
        except Exception as e:
            raise CustomException(e, sys) from e

    def find_best_k(self, X: np.ndarray):
        try:
            cfg = self.config
            inertias, sil_scores = {}, {}
            logger.info(f"Searching K from {cfg.k_min} to {cfg.k_max} ...")
            for k in range(cfg.k_min, cfg.k_max + 1):
                km = KMeans(n_clusters=k, n_init=cfg.n_init, max_iter=cfg.max_iter, random_state=cfg.random_seed)
                labels = km.fit_predict(X)
                inertias[k] = km.inertia_
                sil_scores[k] = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=cfg.random_seed)
                logger.info(f"  K={k:2d} | inertia={inertias[k]:,.0f} | silhouette={sil_scores[k]:.4f}")
            best_k = max(sil_scores, key=sil_scores.get)
            logger.info(f"Best K = {best_k} (silhouette={sil_scores[best_k]:.4f})")
            return best_k, inertias, sil_scores
        except Exception as e:
            raise CustomException(e, sys) from e

    def plot_elbow_silhouette(self, inertias: dict, sil_scores: dict, best_k: int) -> str:
        try:
            ks = sorted(inertias.keys())
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("KMeans Model Selection", fontsize=14, fontweight="bold")

            axes[0].plot(ks, [inertias[k] for k in ks], "b-o", linewidth=2)
            axes[0].axvline(x=best_k, color="red", linestyle="--", label=f"Best K={best_k}")
            axes[0].set_title("Elbow Method (Inertia)")
            axes[0].set_xlabel("Number of Clusters (K)")
            axes[0].set_ylabel("Inertia")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].bar(ks, [sil_scores[k] for k in ks],
                        color=["red" if k == best_k else "steelblue" for k in ks])
            axes[1].set_title("Silhouette Score by K")
            axes[1].set_xlabel("Number of Clusters (K)")
            axes[1].set_ylabel("Silhouette Score")
            axes[1].grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            save_path = str(self.config.artifacts_dir / "elbow_silhouette.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Elbow/Silhouette plot saved to {save_path}")
            return save_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def plot_pca_clusters(self, X_scaled: np.ndarray, labels: np.ndarray) -> str:
        try:
            pca = PCA(n_components=2, random_state=self.config.random_seed)
            X_pca = pca.fit_transform(X_scaled)
            explained = pca.explained_variance_ratio_

            fig, ax = plt.subplots(figsize=(10, 7))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.5, s=10)
            plt.colorbar(scatter, ax=ax, label="Cluster")
            ax.set_title(
                f"PCA Cluster Visualization\n(PC1={explained[0]:.1%}, PC2={explained[1]:.1%} variance explained)",
                fontsize=12,
            )
            ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
            ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_path = str(self.config.artifacts_dir / "pca_clusters.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"PCA visualization saved to {save_path}")
            return save_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def assign_segment_labels(self, df: pd.DataFrame, labels: np.ndarray):
        try:
            df = df.copy()
            df["Cluster"] = labels
            cluster_value = df.groupby("Cluster")["Lifetime_Value"].mean().sort_values(ascending=False)
            label_map = {
                cluster: SEGMENT_NAMES[i % len(SEGMENT_NAMES)]
                for i, cluster in enumerate(cluster_value.index)
            }
            df["Segment"] = df["Cluster"].map(label_map)
            return df, label_map
        except Exception as e:
            raise CustomException(e, sys) from e

    def build_cluster_profiles(self, df: pd.DataFrame) -> str:
        try:
            profile = (
                df[self.config.feature_cols + ["Cluster", "Segment"]]
                .groupby(["Cluster", "Segment"])
                .mean()
                .reset_index()
            )
            save_path = str(self.config.cluster_profiles_path)
            profile.to_csv(save_path, index=False)
            logger.info(f"Cluster profiles saved to {save_path}")
            return save_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def run(self):
        try:
            load_dotenv()
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            mlflow.set_experiment(self.config.experiment_name)

            df, X_scaled = self.load_data()
            best_k, inertias, sil_scores = self.find_best_k(X_scaled)

            logger.info(f"Training final KMeans with K={best_k}")
            cfg = self.config
            kmeans = KMeans(n_clusters=best_k, n_init=cfg.n_init, max_iter=cfg.max_iter, random_state=cfg.random_seed)
            labels = kmeans.fit_predict(X_scaled)

            sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)), random_state=cfg.random_seed)
            ch = calinski_harabasz_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            logger.info(f"Metrics — silhouette={sil:.4f}, calinski_harabasz={ch:.2f}, davies_bouldin={db:.4f}")

            elbow_path = self.plot_elbow_silhouette(inertias, sil_scores, best_k)
            pca_path = self.plot_pca_clusters(X_scaled, labels)
            df, label_map = self.assign_segment_labels(df, labels)
            profiles_path = self.build_cluster_profiles(df)

            bundle = {
                "model": kmeans,
                "label_map": label_map,
                "feature_cols": cfg.feature_cols,
            }
            joblib.dump(bundle, cfg.model_path)
            logger.info(f"Model bundle saved to {cfg.model_path}")

            with mlflow.start_run():
                mlflow.log_params({
                    "k": best_k,
                    "k_min": cfg.k_min,
                    "k_max": cfg.k_max,
                    "n_init": cfg.n_init,
                    "max_iter": cfg.max_iter,
                    "random_seed": cfg.random_seed,
                })
                mlflow.log_metrics({
                    "silhouette_score": sil,
                    "calinski_harabasz_score": ch,
                    "davies_bouldin_score": db,
                })
                mlflow.log_artifact(elbow_path)
                mlflow.log_artifact(pca_path)
                mlflow.log_artifact(profiles_path)
                mlflow.sklearn.log_model(kmeans, "kmeans_model")
                logger.info("MLflow run logged to DagsHub")

            logger.info(f"Segment label map: {label_map}")
        except Exception as e:
            raise CustomException(e, sys) from e
