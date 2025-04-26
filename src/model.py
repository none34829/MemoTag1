"""
Machine learning model for cognitive decline detection.
Implements unsupervised learning approaches to identify patterns indicating cognitive impairment.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Any, Optional, Union
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CognitiveDeclineModel:
    """Model for detecting cognitive decline patterns in voice and text features."""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the model.
        
        Args:
            model_dir: Directory to save/load models from
        """
        self.model_dir = model_dir
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.features_to_exclude = ['file_path', 'transcript']
        self.text_features = None
        self.audio_features = None
        self.combined_pipeline = None
        self.anomaly_detector = None
        self.feature_importance = {}
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features before modeling.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Remove non-numeric columns
        for col in self.features_to_exclude:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])
        
        # Handle missing values
        df_processed = df_processed.fillna(0)
        
        return df_processed
    
    def create_feature_pipeline(self, df: pd.DataFrame) -> Pipeline:
        """
        Create preprocessing pipeline for features.
        
        Args:
            df: Training data
            
        Returns:
            Scikit-learn pipeline
        """
        pipeline = Pipeline([
            ('scaler', RobustScaler()),  # Robust to outliers
            ('pca', PCA(n_components=0.95))  # Keep 95% of variance
        ])
        
        # Fit the pipeline
        pipeline.fit(df)
        
        # Get PCA components to understand feature importance
        pca = pipeline.named_steps['pca']
        scaler = pipeline.named_steps['scaler']
        
        # Get feature importance from PCA
        n_components = pca.n_components_
        feature_names = df.columns
        
        for i in range(n_components):
            # Get the loading factors for each feature for this component
            for j, feature in enumerate(feature_names):
                importance = abs(pca.components_[i, j])
                if feature in self.feature_importance:
                    self.feature_importance[feature] += importance * pca.explained_variance_ratio_[i]
                else:
                    self.feature_importance[feature] = importance * pca.explained_variance_ratio_[i]
        
        # Normalize importance values
        total_importance = sum(self.feature_importance.values())
        if total_importance > 0:
            for feature in self.feature_importance:
                self.feature_importance[feature] /= total_importance
        
        return pipeline
    
    def train_anomaly_detection(self, features: np.ndarray, contamination: float = 0.1) -> IsolationForest:
        """
        Train an anomaly detection model.
        
        Args:
            features: Preprocessed features
            contamination: Expected proportion of outliers
            
        Returns:
            Trained anomaly detection model
        """
        model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(features)
        return model
    
    def train_clustering(self, features: np.ndarray, n_clusters: int = None) -> Tuple[Any, int]:
        """
        Train a clustering model to identify patterns.
        
        Args:
            features: Preprocessed features
            n_clusters: Number of clusters (if None, will be determined automatically)
            
        Returns:
            Tuple of (trained clustering model, optimal number of clusters)
        """
        # If n_clusters is not specified, find the optimal number
        if n_clusters is None:
            sil_scores = []
            max_clusters = min(10, features.shape[0] - 1)  # Ensure we don't exceed sample count
            max_clusters = max(2, max_clusters)  # Ensure at least 2 clusters
            
            for n in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                # Skip if only one cluster is found
                if len(np.unique(cluster_labels)) < 2:
                    continue
                
                silhouette_avg = silhouette_score(features, cluster_labels)
                sil_scores.append((n, silhouette_avg))
            
            # Find the optimal number of clusters
            if sil_scores:
                n_clusters = max(sil_scores, key=lambda x: x[1])[0]
            else:
                n_clusters = 2  # Default if no good clustering is found
        
        # Train the final model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(features)
        
        return kmeans, n_clusters
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the cognitive decline detection model.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training")
        
        try:
            # Preprocess features
            df_processed = self.preprocess_features(df)
            
            if df_processed.empty or df_processed.shape[0] < 3:
                logger.error("Not enough samples for training")
                return {"error": "Not enough samples for training"}
            
            # Create feature pipeline
            logger.info("Creating feature preprocessing pipeline")
            self.combined_pipeline = self.create_feature_pipeline(df_processed)
            
            # Transform data
            transformed_features = self.combined_pipeline.transform(df_processed)
            
            # Train anomaly detection model
            logger.info("Training anomaly detection model")
            self.anomaly_detector = self.train_anomaly_detection(transformed_features)
            
            # Get anomaly scores
            anomaly_scores = -self.anomaly_detector.score_samples(transformed_features)
            
            # Train clustering model if we have enough samples
            if df_processed.shape[0] >= 3:
                logger.info("Training clustering model")
                self.clustering_model, n_clusters = self.train_clustering(transformed_features)
                cluster_labels = self.clustering_model.predict(transformed_features)
            else:
                self.clustering_model = None
                n_clusters = 0
                cluster_labels = np.zeros(df_processed.shape[0])
            
            # Calculate cluster statistics
            cluster_stats = {}
            if self.clustering_model is not None:
                for i in range(n_clusters):
                    cluster_mask = cluster_labels == i
                    cluster_stats[f"cluster_{i}"] = {
                        "size": int(np.sum(cluster_mask)),
                        "avg_anomaly_score": float(np.mean(anomaly_scores[cluster_mask])),
                        "std_anomaly_score": float(np.std(anomaly_scores[cluster_mask]))
                    }
            
            # Save models if model_dir is specified
            if self.model_dir:
                joblib.dump(self.combined_pipeline, os.path.join(self.model_dir, "pipeline.pkl"))
                joblib.dump(self.anomaly_detector, os.path.join(self.model_dir, "anomaly_detector.pkl"))
                if self.clustering_model is not None:
                    joblib.dump(self.clustering_model, os.path.join(self.model_dir, "clustering_model.pkl"))
            
            # Prepare result dictionary
            training_results = {
                "n_samples": df_processed.shape[0],
                "n_features": df_processed.shape[1],
                "n_clusters": n_clusters,
                "top_features": sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10],
                "cluster_stats": cluster_stats
            }
            
            logger.info("Model training completed successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {"error": str(e)}
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with prediction results
        """
        logger.info("Making predictions on new data")
        
        try:
            # Check if models are trained
            if self.combined_pipeline is None or self.anomaly_detector is None:
                logger.info("Models not trained yet, using demo mode")
                # Use demo mode instead of returning an error
                return self._demo_mode_prediction(df)
            
            # Preprocess features
            df_processed = self.preprocess_features(df)
            
            # Transform data
            transformed_features = self.combined_pipeline.transform(df_processed)
            
            # Get anomaly scores (-1 for outliers, 1 for inliers in IsolationForest)
            # We convert so that higher values = more anomalous
            anomaly_scores = -self.anomaly_detector.score_samples(transformed_features)
            
            # Get cluster assignments
            if self.clustering_model is not None:
                cluster_labels = self.clustering_model.predict(transformed_features)
            else:
                cluster_labels = np.zeros(df_processed.shape[0])
            
            # Create results DataFrame
            results = []
            for i in range(df.shape[0]):
                # Get original file path and transcript if available
                file_path = df['file_path'].iloc[i] if 'file_path' in df.columns else f"sample_{i}"
                transcript = df['transcript'].iloc[i] if 'transcript' in df.columns else ""
                
                # Calculate cognitive decline risk score (normalized between 0-100)
                risk_score = min(100, max(0, 50 + 50 * (anomaly_scores[i] - np.mean(anomaly_scores)) / np.std(anomaly_scores))) if len(anomaly_scores) > 1 else 50
                
                results.append({
                    "sample_id": i,
                    "file_path": file_path,
                    "anomaly_score": float(anomaly_scores[i]),
                    "cluster": int(cluster_labels[i]),
                    "risk_score": float(risk_score),
                    "risk_level": "High" if risk_score > 70 else "Medium" if risk_score > 30 else "Low"
                })
            
            # Get feature importance for explanation
            feature_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "predictions": results,
                "key_indicators": feature_importance,
                "average_risk_score": float(np.mean([r["risk_score"] for r in results]))
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {"error": str(e)}
    
    def load_models(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.model_dir or not os.path.exists(self.model_dir):
            logger.error("Model directory not specified or does not exist")
            return False
        
        try:
            # Load pipeline
            pipeline_path = os.path.join(self.model_dir, "pipeline.pkl")
            if os.path.exists(pipeline_path):
                self.combined_pipeline = joblib.load(pipeline_path)
            else:
                logger.error(f"Pipeline model not found at {pipeline_path}")
                return False
            
            # Load anomaly detector
            anomaly_path = os.path.join(self.model_dir, "anomaly_detector.pkl")
            if os.path.exists(anomaly_path):
                self.anomaly_detector = joblib.load(anomaly_path)
            else:
                logger.error(f"Anomaly detector model not found at {anomaly_path}")
                return False
            
            # Load clustering model if exists
            clustering_path = os.path.join(self.model_dir, "clustering_model.pkl")
            if os.path.exists(clustering_path):
                self.clustering_model = joblib.load(clustering_path)
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance rankings.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def generate_report(self, df: pd.DataFrame, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive report of findings.
        
        Args:
            df: Original feature DataFrame
            predictions: Prediction results
            
        Returns:
            Dictionary with report data
        """
        if "error" in predictions:
            return {"error": predictions["error"]}
        
        try:
            # Extract feature correlations with risk scores
            correlations = {}
            risk_scores = np.array([p["risk_score"] for p in predictions["predictions"]])
            
            df_numeric = self.preprocess_features(df)
            
            for column in df_numeric.columns:
                if df_numeric[column].std() > 0:  # Skip constant columns
                    correlation = np.corrcoef(df_numeric[column], risk_scores)[0, 1]
                    correlations[column] = correlation
            
            # Sort by absolute correlation value
            sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Top indicators (based on correlation with risk score)
            top_indicators = [{
                "feature": feature,
                "correlation": round(corr, 3),
                "direction": "Higher values increase risk" if corr > 0 else "Lower values increase risk"
            } for feature, corr in sorted_correlations[:10] if abs(corr) > 0.2]
            
            # Cluster analysis
            prediction_df = pd.DataFrame(predictions["predictions"])
            cluster_analysis = []
            
            if "cluster" in prediction_df.columns and len(prediction_df["cluster"].unique()) > 1:
                for cluster in prediction_df["cluster"].unique():
                    cluster_samples = prediction_df[prediction_df["cluster"] == cluster]
                    cluster_analysis.append({
                        "cluster_id": int(cluster),
                        "sample_count": int(cluster_samples.shape[0]),
                        "avg_risk_score": float(cluster_samples["risk_score"].mean()),
                        "std_risk_score": float(cluster_samples["risk_score"].std()),
                        "high_risk_percentage": float((cluster_samples["risk_score"] > 70).mean() * 100)
                    })
            
            # Overall statistics
            avg_risk = float(prediction_df["risk_score"].mean())
            std_risk = float(prediction_df["risk_score"].std())
            high_risk_percentage = float((prediction_df["risk_score"] > 70).mean() * 100)
            
            # Generate detailed report
            return {
                "overall_statistics": {
                    "sample_count": df.shape[0],
                    "average_risk_score": avg_risk,
                    "risk_score_std": std_risk,
                    "high_risk_percentage": high_risk_percentage
                },
                "key_indicators": top_indicators,
                "cluster_analysis": cluster_analysis,
                "feature_importance": self.get_feature_importance()
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"error": str(e)}
    
    def _demo_mode_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate demo prediction results when model isn't trained yet.
        This allows the API to function without sufficient training data.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with simulated prediction results
        """
        logger.info("Using demo mode for predictions")
        
        # Basic feature analysis for pattern detection
        features_for_analysis = {}
        
        # Extract and analyze key features if they exist in the dataframe
        for sample_idx, row in df.iterrows():
            # Get audio features if available
            audio_features = {}
            for col in df.columns:
                if col not in self.features_to_exclude:
                    try:
                        audio_features[col] = float(row[col])
                    except (ValueError, TypeError):
                        pass
            
            # Generate risk score based on key cognitive decline indicators
            # For demo, we'll use certain well-known indicators
            risk_score = 50.0  # Default baseline score
            
            # 1. Analyze pause patterns if available
            if 'pause_count' in audio_features and 'total_duration' in audio_features:
                pause_rate = audio_features.get('pause_count', 0) / max(audio_features.get('total_duration', 1), 1)
                if pause_rate > 0.5:  # High pause rate
                    risk_score += 15
            
            # 2. Check pitch variability
            if 'pitch_variability_coefficient' in audio_features:
                pitch_var = audio_features.get('pitch_variability_coefficient', 0)
                if pitch_var < 0.1:  # Low pitch variability indicates monotone speech
                    risk_score += 10
            
            # 3. Check speech rate
            if 'speech_rate' in audio_features:
                speech_rate = audio_features.get('speech_rate', 0)
                if speech_rate < 2.0:  # Slower speech rate
                    risk_score += 10
            
            # 4. Check hesitation markers if available
            if 'hesitation_marker_count' in audio_features:
                hesitation_count = audio_features.get('hesitation_marker_count', 0)
                if hesitation_count > 5:  # High number of hesitations
                    risk_score += 15
            
            # Cap the risk score at 0-100
            risk_score = max(0, min(100, risk_score))
            
            # Determine risk level based on score
            if risk_score < 30:
                risk_level = "Low"
            elif risk_score < 70:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            
            # Generate simulated feature importance
            feature_importance = [
                {"feature": "pause_patterns", "importance": 0.35},
                {"feature": "speech_rhythm", "importance": 0.25},
                {"feature": "hesitation_markers", "importance": 0.20},
                {"feature": "pitch_variability", "importance": 0.15},
                {"feature": "spectral_features", "importance": 0.05}
            ]
            
            # Return prediction results in the expected format
            return {
                "risk_scores": [risk_score],
                "risk_levels": [risk_level],
                "anomaly_scores": [risk_score / 100.0],  # Normalized to 0-1
                "feature_importance": feature_importance,
                "demo_mode": True  # Flag to indicate these are demo results
            }
    
    def get_risk_score(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Get risk score for a single sample.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Dictionary with risk score and explanation
        """
        try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])
            
            # Get predictions
            predictions = self.predict(df)
            
            if "error" in predictions:
                return {"error": predictions["error"]}
            
            # Extract result for the single sample
            result = predictions["predictions"][0]
            
            # Add explanations based on feature importance
            explanations = []
            
            for feature, importance in predictions["key_indicators"]:
                if feature in features:
                    explanations.append({
                        "feature": feature,
                        "importance": float(importance),
                        "value": float(features.get(feature, 0))
                    })
            
            return {
                "risk_score": result["risk_score"],
                "risk_level": result["risk_level"],
                "explanations": explanations
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return {"error": str(e)}
