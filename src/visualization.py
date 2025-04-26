"""
Visualization module for cognitive decline detection.
Creates insightful visualizations of voice features, text patterns, and ML results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """Class for generating visualizations for cognitive decline analysis."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the visualization generator.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set default plot style
        sns.set(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12
    
    def save_or_show(self, fig, filename: str = None):
        """
        Save figure to file or display it.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
        """
        if filename and self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close(fig)
            logger.info(f"Saved visualization to {filepath}")
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_feature_distribution(self, df: pd.DataFrame, feature_columns: List[str], 
                                  filename: str = None) -> plt.Figure:
        """
        Plot distribution of selected features.
        
        Args:
            df: DataFrame with features
            feature_columns: List of columns to plot
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        n_features = len(feature_columns)
        if n_features == 0:
            logger.warning("No features to plot")
            return None
        
        # Determine subplot grid dimensions
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Flatten axes array for easier iteration
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        
        for i, feature in enumerate(feature_columns):
            if i < len(axes):
                ax = axes[i]
                if feature in df.columns:
                    sns.histplot(df[feature], kde=True, ax=ax)
                    ax.set_title(f"Distribution of {feature}")
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Frequency")
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        self.save_or_show(fig, filename)
        return fig
    
    def plot_feature_correlations(self, df: pd.DataFrame, target_column: str = None, 
                                 filename: str = None) -> plt.Figure:
        """
        Plot correlation matrix of features.
        
        Args:
            df: DataFrame with features
            target_column: Optional target column to highlight
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        # Remove non-numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            logger.warning("No numeric features to plot correlations")
            return None
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Use colormap with high contrast for better readability
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f", ax=ax)
        
        # Highlight target column if specified
        if target_column and target_column in corr_matrix.columns:
            target_corr = corr_matrix[target_column].sort_values(ascending=False)
            top_features = target_corr[target_corr.index != target_column].head(10)
            
            plt.figure(figsize=(10, 8))
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            sns.barplot(x=top_features.values, y=top_features.index, ax=ax2)
            ax2.set_title(f"Top Features Correlated with {target_column}")
            ax2.set_xlabel("Correlation")
            
            self.save_or_show(fig2, f"top_correlations_{filename}" if filename else None)
        
        plt.title("Feature Correlation Matrix")
        self.save_or_show(fig, filename)
        return fig
    
    def plot_dimensionality_reduction(self, df: pd.DataFrame, labels: List[int] = None,
                                     method: str = 'pca', filename: str = None) -> plt.Figure:
        """
        Plot dimensionality reduction of features (PCA or t-SNE).
        
        Args:
            df: DataFrame with features
            labels: Optional cluster labels
            method: 'pca' or 'tsne'
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        # Remove non-numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            logger.warning("No numeric features for dimensionality reduction")
            return None
        
        if numeric_df.shape[1] < 2:
            logger.warning("Need at least 2 features for dimensionality reduction")
            return None
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            model = PCA(n_components=2)
            title = "PCA Projection of Features"
        else:  # t-SNE
            model = TSNE(n_components=2, perplexity=min(30, max(5, numeric_df.shape[0]//5)),
                        random_state=42)
            title = "t-SNE Projection of Features"
        
        # Transform data
        transformed = model.fit_transform(numeric_df)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot with or without labels
        if labels is not None:
            scatter = ax.scatter(transformed[:, 0], transformed[:, 1], c=labels, 
                                cmap='viridis', alpha=0.8, s=100)
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend)
        else:
            ax.scatter(transformed[:, 0], transformed[:, 1], alpha=0.8, s=100)
        
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        
        # Add sample indices as annotations
        for i, (x, y) in enumerate(transformed):
            ax.annotate(str(i), (x, y), fontsize=9, alpha=0.7)
        
        self.save_or_show(fig, filename)
        return fig
    
    def plot_risk_scores(self, predictions: List[Dict[str, Any]], 
                         filename: str = None) -> plt.Figure:
        """
        Plot risk scores from model predictions.
        
        Args:
            predictions: List of prediction dictionaries
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        # Extract risk scores and sample IDs
        risk_scores = [pred["risk_score"] for pred in predictions]
        sample_ids = [pred.get("sample_id", i) for i, pred in enumerate(predictions)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create color map based on risk level
        colors = ['green' if score < 30 else 'orange' if score < 70 else 'red' 
                 for score in risk_scores]
        
        # Plot horizontal bar chart
        bars = ax.barh(sample_ids, risk_scores, color=colors)
        
        # Add risk level threshold lines
        ax.axvline(x=30, color='orange', linestyle='--', alpha=0.7, label='Medium Risk Threshold')
        ax.axvline(x=70, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
        
        # Add labels and title
        ax.set_title("Cognitive Decline Risk Scores by Sample")
        ax.set_xlabel("Risk Score (0-100)")
        ax.set_ylabel("Sample ID")
        ax.legend()
        
        # Add value labels to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 1
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                   va='center')
        
        self.save_or_show(fig, filename)
        return fig
    
    def plot_top_features(self, feature_importance: Dict[str, float], 
                        n_features: int = 10, filename: str = None) -> plt.Figure:
        """
        Plot top important features from the model.
        
        Args:
            feature_importance: Dictionary mapping features to importance scores
            n_features: Number of top features to show
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_n = sorted_features[:n_features]
        
        # Unpack feature names and importance scores
        feature_names = [item[0] for item in top_n]
        importance_scores = [item[1] for item in top_n]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot horizontal bar chart
        bars = ax.barh(feature_names, importance_scores, color='skyblue')
        
        # Add values to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                   va='center')
        
        # Add labels and title
        ax.set_title("Top Features for Cognitive Decline Detection")
        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Feature")
        
        # Reverse y-axis to show most important feature at the top
        ax.invert_yaxis()
        
        self.save_or_show(fig, filename)
        return fig
    
    def plot_cluster_analysis(self, cluster_data: List[Dict[str, Any]], 
                             filename: str = None) -> plt.Figure:
        """
        Plot cluster analysis results.
        
        Args:
            cluster_data: List of cluster statistics
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        if not cluster_data:
            logger.warning("No cluster data to plot")
            return None
        
        # Create DataFrame from cluster data
        df = pd.DataFrame(cluster_data)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Cluster sizes
        axes[0, 0].bar(df["cluster_id"], df["sample_count"], color='skyblue')
        axes[0, 0].set_title("Samples per Cluster")
        axes[0, 0].set_xlabel("Cluster ID")
        axes[0, 0].set_ylabel("Number of Samples")
        
        # Plot 2: Average risk scores by cluster
        bars = axes[0, 1].bar(df["cluster_id"], df["avg_risk_score"], 
                             yerr=df["std_risk_score"], capsize=5,
                             color=['green' if score < 30 else 'orange' if score < 70 else 'red' 
                                   for score in df["avg_risk_score"]])
        axes[0, 1].set_title("Average Risk Score by Cluster")
        axes[0, 1].set_xlabel("Cluster ID")
        axes[0, 1].set_ylabel("Risk Score (0-100)")
        axes[0, 1].axhline(y=30, color='orange', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(y=70, color='red', linestyle='--', alpha=0.7)
        
        # Plot 3: High risk percentage by cluster
        axes[1, 0].bar(df["cluster_id"], df["high_risk_percentage"], color='salmon')
        axes[1, 0].set_title("Percentage of High Risk Samples by Cluster")
        axes[1, 0].set_xlabel("Cluster ID")
        axes[1, 0].set_ylabel("High Risk %")
        
        # Plot 4: Risk score distribution by cluster (boxplot)
        # For this we would need individual sample data, which we don't have in the aggregate data
        # Instead, show a placeholder or alternative visualization
        axes[1, 1].text(0.5, 0.5, "Risk Score Distribution\n(Requires individual sample data)",
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title("Risk Score Distribution by Cluster")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        self.save_or_show(fig, filename)
        return fig
    
    def plot_combined_features(self, df: pd.DataFrame, feature_groups: Dict[str, List[str]],
                              filename: str = None) -> plt.Figure:
        """
        Plot combined features by category (audio, linguistic, etc.).
        
        Args:
            df: DataFrame with features
            feature_groups: Dictionary mapping categories to feature lists
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        n_groups = len(feature_groups)
        if n_groups == 0:
            logger.warning("No feature groups to plot")
            return None
        
        # Create figure
        fig, axes = plt.subplots(n_groups, 1, figsize=(14, 6*n_groups))
        
        # If only one group, wrap axes in list
        if n_groups == 1:
            axes = [axes]
        
        # Plot each group
        for i, (group_name, features) in enumerate(feature_groups.items()):
            ax = axes[i]
            
            # Filter valid features that exist in the DataFrame
            valid_features = [f for f in features if f in df.columns]
            
            if not valid_features:
                ax.text(0.5, 0.5, f"No valid features in group: {group_name}",
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f"{group_name} Features")
                ax.axis('off')
                continue
            
            # Create a boxplot for the group
            sub_df = df[valid_features]
            
            # Scale features for better visualization
            scaled_df = (sub_df - sub_df.min()) / (sub_df.max() - sub_df.min())
            
            # Plot boxplot
            sns.boxplot(data=scaled_df, ax=ax)
            
            # Add jittered points for all samples
            for feature in valid_features:
                sns.stripplot(x=feature, y=scaled_df[feature], 
                             size=4, color='black', ax=ax, alpha=0.5)
            
            ax.set_title(f"{group_name} Features (Normalized Scale)")
            ax.set_ylabel("Normalized Value")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        self.save_or_show(fig, filename)
        return fig
    
    def create_summary_dashboard(self, df: pd.DataFrame, predictions: Dict[str, Any],
                               report: Dict[str, Any], filename: str = None) -> plt.Figure:
        """
        Create a summary dashboard with key visualizations.
        
        Args:
            df: Original feature DataFrame
            predictions: Model predictions
            report: Analysis report
            filename: Output filename
            
        Returns:
            Matplotlib figure
        """
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define grid layout
        grid = plt.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Risk score distribution
        ax1 = fig.add_subplot(grid[0, 0:2])
        risk_scores = [p["risk_score"] for p in predictions["predictions"]]
        sample_ids = [p.get("sample_id", i) for i, p in enumerate(predictions["predictions"])]
        colors = ['green' if score < 30 else 'orange' if score < 70 else 'red' 
                 for score in risk_scores]
        ax1.barh(sample_ids, risk_scores, color=colors)
        ax1.axvline(x=30, color='orange', linestyle='--', alpha=0.7)
        ax1.axvline(x=70, color='red', linestyle='--', alpha=0.7)
        ax1.set_title("Cognitive Decline Risk Scores")
        ax1.set_xlabel("Risk Score (0-100)")
        ax1.set_ylabel("Sample ID")
        
        # 2. Top features importance
        ax2 = fig.add_subplot(grid[0, 2])
        if "feature_importance" in report:
            # Get top 5 features
            features = list(report["feature_importance"].items())[:5]
            feature_names = [item[0] for item in features]
            importance_scores = [item[1] for item in features]
            
            ax2.barh(feature_names, importance_scores, color='skyblue')
            ax2.set_title("Top 5 Important Features")
            ax2.set_xlabel("Relative Importance")
            ax2.invert_yaxis()  # Most important at top
        else:
            ax2.text(0.5, 0.5, "Feature importance data not available",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
            ax2.set_title("Feature Importance")
            ax2.axis('off')
        
        # 3. Dimensionality reduction (PCA)
        ax3 = fig.add_subplot(grid[1, 0:2])
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.shape[1] >= 2:
            # Add cluster labels if available
            if predictions["predictions"] and "cluster" in predictions["predictions"][0]:
                labels = [p["cluster"] for p in predictions["predictions"]]
            else:
                labels = None
            
            # Apply PCA
            pca = PCA(n_components=2)
            transformed = pca.fit_transform(numeric_df)
            
            # Plot with or without labels
            if labels:
                scatter = ax3.scatter(transformed[:, 0], transformed[:, 1], c=labels, 
                                    cmap='viridis', s=100, alpha=0.8)
                legend = ax3.legend(*scatter.legend_elements(), title="Clusters")
                ax3.add_artist(legend)
            else:
                ax3.scatter(transformed[:, 0], transformed[:, 1], s=100, alpha=0.8)
            
            # Add sample indices as annotations
            for i, (x, y) in enumerate(transformed):
                ax3.annotate(str(i), (x, y), fontsize=9, alpha=0.7)
            
            ax3.set_title("PCA Projection of Features")
            ax3.set_xlabel("Component 1")
            ax3.set_ylabel("Component 2")
        else:
            ax3.text(0.5, 0.5, "Not enough numeric features for PCA",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes)
            ax3.set_title("PCA Projection")
            ax3.axis('off')
        
        # 4. Key indicators correlation with risk
        ax4 = fig.add_subplot(grid[1, 2])
        if "key_indicators" in report:
            indicators = report["key_indicators"][:5] if len(report["key_indicators"]) > 0 else []
            
            if indicators:
                features = [ind["feature"] for ind in indicators]
                correlations = [ind["correlation"] for ind in indicators]
                
                bars = ax4.barh(features, correlations, 
                              color=['red' if c < 0 else 'blue' for c in correlations])
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    label_x_pos = width + 0.05 if width >= 0 else width - 0.05
                    ha = 'left' if width >= 0 else 'right'
                    ax4.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                           f'{width:.2f}', ha=ha, va='center')
                
                ax4.set_title("Feature Correlation with Risk")
                ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax4.set_xlim(-1, 1)
                ax4.invert_yaxis()  # Highest correlation at top
            else:
                ax4.text(0.5, 0.5, "No key indicators data available",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes)
                ax4.set_title("Feature Correlation with Risk")
                ax4.axis('off')
        else:
            ax4.text(0.5, 0.5, "Key indicators data not available",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes)
            ax4.set_title("Feature Correlation with Risk")
            ax4.axis('off')
        
        # 5. Summary statistics table
        ax5 = fig.add_subplot(grid[2, 0])
        ax5.axis('tight')
        ax5.axis('off')
        
        if "overall_statistics" in report:
            stats = report["overall_statistics"]
            table_data = [
                ["Metric", "Value"],
                ["Sample Count", stats.get("sample_count", "N/A")],
                ["Avg Risk Score", f"{stats.get('average_risk_score', 'N/A'):.2f}"],
                ["Risk Score Std", f"{stats.get('risk_score_std', 'N/A'):.2f}"],
                ["High Risk %", f"{stats.get('high_risk_percentage', 'N/A'):.1f}%"]
            ]
            
            table = ax5.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
            
            # Style the header row
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(color='white')
            
            ax5.set_title("Summary Statistics")
        else:
            ax5.text(0.5, 0.5, "Summary statistics not available",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax5.transAxes)
            ax5.set_title("Summary Statistics")
        
        # 6. Cluster analysis table
        ax6 = fig.add_subplot(grid[2, 1:3])
        ax6.axis('tight')
        ax6.axis('off')
        
        if "cluster_analysis" in report and report["cluster_analysis"]:
            clusters = report["cluster_analysis"]
            
            table_data = [["Cluster ID", "Sample Count", "Avg Risk Score", "High Risk %"]]
            
            for cluster in clusters:
                table_data.append([
                    cluster.get("cluster_id", "N/A"),
                    cluster.get("sample_count", "N/A"),
                    f"{cluster.get('avg_risk_score', 'N/A'):.2f}",
                    f"{cluster.get('high_risk_percentage', 'N/A'):.1f}%"
                ])
            
            table = ax6.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
            
            # Style the header row
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(color='white')
            
            ax6.set_title("Cluster Analysis")
        else:
            ax6.text(0.5, 0.5, "Cluster analysis data not available",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax6.transAxes)
            ax6.set_title("Cluster Analysis")
        
        # Add main title
        fig.suptitle("Cognitive Decline Detection Analysis Dashboard", fontsize=20, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for main title
        self.save_or_show(fig, filename)
        return fig
