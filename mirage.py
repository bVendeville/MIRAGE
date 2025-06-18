import subprocess
import sys
import importlib.util
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import spacy
import warnings
from typing import Union

from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    log_loss, roc_curve, auc, precision_recall_curve, confusion_matrix
)


from datasets import DatasetDict, Dataset, load_dataset
import datasets

from transformers import T5ForConditionalGeneration, PreTrainedModel
from transformers import T5Tokenizer, PreTrainedTokenizer, AutoTokenizer
from transformers import pipeline
from transformers import EvalPrediction

# Optional: for interactive plotting
import plotly.express as px
import plotly.graph_objects as go


#------------------------------------
# Utils
#------------------------------------

def check_and_install(package: str) -> None:
    # Check if the package is installed
    package_spec = importlib.util.find_spec(package)
    if package_spec is None:
        # Package not found, install it
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        print(f"{package} is already installed.")


def OLD_load_to_hf_dataset(obj):
    if isinstance(obj, pd.DataFrame):
        # Convert pandas DataFrame to Hugging Face Dataset
        return Dataset.from_pandas(obj)
    elif isinstance(obj, Dataset):
        # Already a Hugging Face Dataset
        return obj
    elif isinstance(obj, str):
        # Check if it's a path
        if not os.path.exists(obj):
            raise FileNotFoundError(f"Path '{obj}' does not exist.")
        
        file_ext = os.path.splitext(obj)[-1].lower()
        if file_ext == ".json":
            # Load JSON file
            print(pd.DataFrame(obj))
            return Dataset.from_json(obj)
        elif file_ext in [".csv", ".xls", ".xlsx"]:
            # Load CSV or Excel file into pandas first
            df = pd.read_csv(obj) if file_ext == ".csv" else pd.read_excel(obj)
            return Dataset.from_pandas(df)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    else:
        raise TypeError("Unsupported input type. Expected pandas.DataFrame, Hugging Face Dataset, or file path.")

def load_to_hf_dataset(data: Union[str, pd.DataFrame, Dataset]) -> Dataset:
    """
    Ensure input is a Hugging Face Dataset, converting from path or DataFrame if needed.

    Args:
        data (str | pd.DataFrame | Dataset): Path to file (csv/json), pandas DataFrame, or Dataset.

    Returns:
        Dataset: Hugging Face Dataset
    """
    if isinstance(data, Dataset):
        return data
    elif isinstance(data, pd.DataFrame):
        return Dataset.from_pandas(data)
    elif isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"File not found: {data}")
        
        ext = os.path.splitext(data)[1].lower()
        if ext == ".csv":
            return load_dataset("csv", data_files=data)["train"]
        elif ext == ".json":
            return load_dataset("json", data_files=data)["train"]
        elif ext == ".txt":
            return load_dataset("text", data_files=data)["train"]
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    else:
        raise TypeError(f"Unsupported input type: {type(data)}. Must be Dataset, DataFrame, or path to file.")
    
# Usage examples:
# df = pd.DataFrame(...)  # Example DataFrame
# ds = load_to_hf_dataset(df)

# hf_dataset = Dataset.from_dict({"col1": [1, 2], "col2": ["a", "b"]})
# ds = load_to_hf_dataset(hf_dataset)

# ds = load_to_hf_dataset("data.json")


#------------------------------------
# Class to generate accuracy, recall etc... and graphs on annotated evaluated datasets
#------------------------------------

# Handle datasets import
try:
    from datasets import Dataset, DatasetDict
except ImportError:
    # Fallback for when datasets is not available
    Dataset = dict
    DatasetDict = dict


class ScoreManager:
    def __init__(self, dataset, metric_list: list[str], on_split: str = "test"):
        """
        Initialize the ScoreManager with dataset and metrics to compute.

        Args:
            dataset: The dataset containing 'label' and 'predictions' keys.
            metric_list (list): List of metric names to compute. Supported metrics:
                                ['f1', 'precision', 'recall', 'accuracy',
                                 'balanced_accuracy', 'mcc', 'kappa',
                                 'log_loss', 'roc_values', 'auroc', 'auprc',
                                 'confusion_matrix', 'precision_recall_values', 'data_stats'].
            on_split (str): Dataset split to use if dataset has multiple splits.
        """
        # Handle dataset splits
        self.dataset = self._handle_dataset_splits(dataset, on_split)
        
        # Extract labels and predictions
        self.labels = self._extract_labels()
        self.class_scores = self.dataset["predictions"]
        self.predicted_labels = [self._get_predicted_label(example) for example in self.dataset]
        self.metric_list = metric_list
        
        # Initialize results dictionary
        self.results = {}
        
        # Set label names for confusion matrix
        self.label_names = ['HALL', 'NOHALL']
        
        # Calculate all requested metrics
        self._calculate_metrics()

    def _handle_dataset_splits(self, dataset, on_split):
        """Handle dataset with or without splits."""
        if hasattr(dataset, 'keys') and callable(dataset.keys):
            # Check if it's a DatasetDict-like object
            if on_split is None:
                raise ValueError("The dataset has splits, but no split was provided. Please specify a split (e.g., 'train', 'test').")
            
            if on_split in dataset:
                return dataset[on_split]
            else:
                available_splits = list(dataset.keys()) if hasattr(dataset, 'keys') else ['unknown']
                raise ValueError(f"Split '{on_split}' does not exist in the dataset. Available splits: {available_splits}")
        
        return dataset

    def _extract_labels(self):
        """Extract and flatten labels if needed."""
        labels = self.dataset["label"]
        
        # Flatten labels if they are nested lists
        if isinstance(labels[0], list):
            labels = [label[0] for label in labels]
        
        return labels

    def _get_predicted_label(self, example):
        """Get the label with the highest score from predictions."""
        prediction_dict = example["predictions"]
        return max(prediction_dict, key=prediction_dict.get)

    def _calculate_metrics(self):
        """Calculate and store all specified metrics."""
        for metric in self.metric_list:
            if metric == "data_stats":
                self.results["data_stats"] = self._calculate_data_stats()
            else:
                method_name = f"_calculate_{metric}"
                if hasattr(self, method_name):
                    self.results[metric] = getattr(self, method_name)()
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

    # Metric calculation methods
    def _calculate_f1(self):
        """Calculate F1 score."""
        return f1_score(self.labels, self.predicted_labels, average='weighted')

    def _calculate_precision(self):
        """Calculate precision score."""
        return precision_score(self.labels, self.predicted_labels, average='weighted')

    def _calculate_recall(self):
        """Calculate recall score."""
        return recall_score(self.labels, self.predicted_labels, average='weighted')

    def _calculate_accuracy(self):
        """Calculate accuracy score."""
        return accuracy_score(self.labels, self.predicted_labels)

    def _calculate_balanced_accuracy(self):
        """Calculate balanced accuracy score."""
        return balanced_accuracy_score(self.labels, self.predicted_labels)

    def _calculate_mcc(self):
        """Calculate Matthews Correlation Coefficient."""
        return matthews_corrcoef(self.labels, self.predicted_labels)

    def _calculate_kappa(self):
        """Calculate Cohen's Kappa score."""
        return cohen_kappa_score(self.labels, self.predicted_labels)

    def _calculate_log_loss(self):
        """Calculate Log Loss (Cross-Entropy Loss)."""
        y_true = self.labels
        class_names = sorted(list(self.class_scores[0].keys()))
        y_pred_proba = [[pred[class_name] for class_name in class_names] for pred in self.class_scores]
        return log_loss(y_true, y_pred_proba, labels=class_names)

    def _calculate_roc_values(self):
        """Calculate ROC curve values."""
        y_true = np.array([1 if label == 'HALL' else 0 for label in self.labels])
        y_scores = [example['HALL'] for example in self.class_scores]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

    def _calculate_auroc(self):
        """Calculate AUROC (Area Under ROC Curve)."""
        y_true = np.array([1 if label == 'HALL' else 0 for label in self.labels])
        y_scores = [example['HALL'] for example in self.class_scores]
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        return auc(fpr, tpr)

    def _calculate_auprc(self):
        """Calculate area under the precision-recall curve (AUPRC)."""
        y_true = np.array([1 if label == 'HALL' else 0 for label in self.labels])
        y_scores = [example['HALL'] for example in self.class_scores]
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return auc(recall, precision)

    def _calculate_precision_recall_values(self):
        """Calculate Precision-Recall curve values."""
        y_true = np.array([1 if label == 'HALL' else 0 for label in self.labels])
        y_scores = [example['HALL'] for example in self.class_scores]
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        return {'precision': precision, 'recall': recall, 'thresholds': thresholds}

    def _calculate_confusion_matrix(self):
        """Calculate confusion matrix."""
        return confusion_matrix(self.labels, self.predicted_labels, labels=self.label_names)

    def _calculate_data_stats(self):
        """Calculate basic statistics on labels and scores."""
        total = len(self.labels)
        pos_count = sum(1 for l in self.labels if l == 'HALL')
        neg_count = total - pos_count
        pos_ratio = pos_count / total if total else 0
        neg_ratio = neg_count / total if total else 0

        # Scores for positive class
        y_scores = [example['HALL'] for example in self.class_scores]
        mean_score = sum(y_scores) / total if total else 0
        std_score = (sum((s - mean_score) ** 2 for s in y_scores) / total) ** 0.5 if total else 0

        return {
            'total_samples': total,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'pos_ratio': pos_ratio,
            'neg_ratio': neg_ratio,
            'mean_score': mean_score,
            'std_score': std_score
        }

    # Plotting helper methods
    def _ensure_fig_ax(self, ax=None, figsize=(8, 6)):
        """Helper to ensure we have a figure and axes."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            return fig, ax, True
        else:
            return ax.figure, ax, False

    def _save_plot(self, fig, output_path, filename, save_format):
        """Save plot to file."""
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            filepath = os.path.join(output_path, f"{filename}.{save_format}")
            
            if save_format.lower() == 'html':
                # For interactive HTML saving with matplotlib, use mpld3
                try:
                    import mpld3
                    # Configure mpld3 for better interactivity
                    mpld3.enable_notebook()
                    html_str = mpld3.fig_to_html(fig, template_type='general')
                    with open(filepath, 'w') as f:
                        f.write(html_str)
                    print(f"Interactive HTML plot saved: {filepath}")
                except ImportError:
                    print(f"Warning: mpld3 not available. Cannot save {filename} as interactive HTML.")
                    print("Install with: pip install mpld3")
                    print("Falling back to PNG format.")
                    filepath = os.path.join(output_path, f"{filename}.png")
                    fig.savefig(filepath, bbox_inches='tight', dpi=300)
                    print(f"Static plot saved: {filepath}")
            else:
                fig.savefig(filepath, bbox_inches='tight', dpi=300)
                print(f"Plot saved: {filepath}")

    def _save_plotly_figure(self, fig, output_path, filename, save_format):
        """Save plotly figure to file."""
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            filepath = os.path.join(output_path, f"{filename}.{save_format}")
            
            if save_format.lower() == 'html':
                # Save as fully interactive HTML
                fig.write_html(filepath, 
                              include_plotlyjs=True,  # Include plotly.js for full interactivity
                              config={'displayModeBar': True,  # Show toolbar
                                     'toImageButtonOptions': {'format': 'png', 'filename': filename},
                                     'displaylogo': False})
                print(f"Interactive HTML plot saved: {filepath}")
            else:
                try:
                    fig.write_image(filepath)
                    print(f"Plot saved: {filepath}")
                except Exception as e:
                    print(f"Error saving image: {e}")
                    print("For image export, install: pip install kaleido")

    # Individual plotting methods
    def _plot_metrics_bar(self, metrics_bar, show_plot, save_plot, output_path, save_format, interactive, palette, ax=None):
        """Plot metrics bar chart."""
        values = [self.results.get(m, 0) for m in metrics_bar]
        
        if interactive:
            try:
                import plotly.express as px
                import pandas as pd
                
                df = pd.DataFrame({'Metrics': metrics_bar, 'Values': values})
                fig = px.bar(
                    df,
                    x='Metrics',
                    y='Values',
                    title='Metrics Bar Chart',
                    color='Metrics',
                    color_discrete_sequence=palette or px.colors.qualitative.Pastel
                )
                fig.update_layout(showlegend=False)
                
                if save_plot:
                    self._save_plotly_figure(fig, output_path, "metrics_bar", save_format)
                if show_plot:
                    fig.show()
                    
            except ImportError:
                print("Warning: Plotly not available. Using matplotlib instead.")
                interactive = False
        
        if not interactive:
            fig, ax, created = self._ensure_fig_ax(ax)
            
            colors = palette if palette else plt.cm.Set3.colors
            bars = ax.bar(metrics_bar, values, color=colors[:len(metrics_bar)])
            
            ax.set_title("Metrics Bar Chart")
            ax.set_ylabel("Values")
            ax.set_xlabel("Metrics")
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            if save_plot:
                self._save_plot(fig, output_path, "metrics_bar", save_format)
            if show_plot and created:
                plt.show()

    def _plot_roc_curve(self, show_plot, save_plot, output_path, save_format, interactive, ax=None):
        """Plot ROC curve."""
        if 'roc_values' not in self.results:
            self.results['roc_values'] = self._calculate_roc_values()
        if 'auroc' not in self.results:
            self.results['auroc'] = self._calculate_auroc()
            
        roc_values = self.results['roc_values']
        auc_score = self.results['auroc']
        
        if interactive:
            try:
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=roc_values['fpr'].tolist(), 
                    y=roc_values['tpr'].tolist(), 
                    mode='lines',
                    name=f'ROC curve (AUC = {auc_score:.3f})',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], 
                    y=[0, 1], 
                    mode='lines',
                    name='Random classifier',
                    line=dict(dash='dash', color='red')
                ))
                fig.update_layout(
                    title='Receiver Operating Characteristic (ROC) Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    showlegend=True
                )
                
                if save_plot:
                    self._save_plotly_figure(fig, output_path, "roc_curve", save_format)
                if show_plot:
                    fig.show()
                    
            except ImportError:
                print("Warning: Plotly not available. Using matplotlib instead.")
                interactive = False
        
        if not interactive:
            fig, ax, created = self._ensure_fig_ax(ax)
            
            ax.plot(roc_values['fpr'], roc_values['tpr'], 
                   color='blue', linewidth=2, label=f'ROC curve (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], linestyle='--', color='red', 
                   linewidth=1, label='Random classifier')
            
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
            if save_plot:
                self._save_plot(fig, output_path, "roc_curve", save_format)
            if show_plot and created:
                plt.show()

    def _plot_precision_recall_curve(self, show_plot, save_plot, output_path, save_format, interactive, ax=None):
        """Plot Precision-Recall curve."""
        if 'precision_recall_values' not in self.results:
            self.results['precision_recall_values'] = self._calculate_precision_recall_values()
        if 'auprc' not in self.results:
            self.results['auprc'] = self._calculate_auprc()
            
        pr_values = self.results['precision_recall_values']
        auprc_score = self.results['auprc']
        
        if interactive:
            try:
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pr_values['recall'].tolist(), 
                    y=pr_values['precision'].tolist(), 
                    mode='lines',
                    name=f'PR curve (AUC = {auprc_score:.3f})',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision'
                )
                
                if save_plot:
                    self._save_plotly_figure(fig, output_path, "precision_recall_curve", save_format)
                if show_plot:
                    fig.show()
                    
            except ImportError:
                print("Warning: Plotly not available. Using matplotlib instead.")
                interactive = False
        
        if not interactive:
            fig, ax, created = self._ensure_fig_ax(ax)
            
            ax.plot(pr_values['recall'], pr_values['precision'], 
                   color='blue', linewidth=2, label=f'PR curve (AUC = {auprc_score:.3f})')
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_plot:
                self._save_plot(fig, output_path, "precision_recall_curve", save_format)
            if show_plot and created:
                plt.show()

    def _plot_confusion_matrix(self, show_plot, save_plot, output_path, save_format, interactive, ax=None):
        """Plot confusion matrix."""
        if 'confusion_matrix' not in self.results:
            self.results['confusion_matrix'] = self._calculate_confusion_matrix()
            
        cm = self.results['confusion_matrix']
        
        if interactive:
            try:
                import plotly.graph_objects as go
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm.tolist(),
                    x=self.label_names,
                    y=self.label_names,
                    colorscale='Blues',
                    showscale=True,
                    text=cm.tolist(),
                    texttemplate="%{text}",
                    textfont={"size": 16}
                ))
                fig.update_layout(
                    title='Confusion Matrix',
                    xaxis_title='Predicted Label',
                    yaxis_title='True Label'
                )
                
                if save_plot:
                    self._save_plotly_figure(fig, output_path, "confusion_matrix", save_format)
                if show_plot:
                    fig.show()
                    
            except ImportError:
                print("Warning: Plotly not available. Using matplotlib instead.")
                interactive = False
        
        if not interactive:
            fig, ax, created = self._ensure_fig_ax(ax)
            
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            
            ax.set_xticks(range(len(self.label_names)))
            ax.set_yticks(range(len(self.label_names)))
            ax.set_xticklabels(self.label_names)
            ax.set_yticklabels(self.label_names)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=14)
            
            if save_plot:
                self._save_plot(fig, output_path, "confusion_matrix", save_format)
            if show_plot and created:
                plt.show()

    def _plot_score_distribution(self, show_plot, save_plot, output_path, save_format, interactive, palette, ax=None):
        """Plot score distribution."""
        y_scores = [example['HALL'] for example in self.class_scores]
        
        if interactive:
            try:
                import plotly.express as px
                import pandas as pd
                
                df = pd.DataFrame({'scores': y_scores})
                fig = px.histogram(
                    df,
                    x='scores',
                    nbins=50,
                    title='Distribution of HALL Scores',
                    labels={'scores': 'Score', 'count': 'Count'},
                    color_discrete_sequence=palette or ['blue']
                )
                
                if save_plot:
                    self._save_plotly_figure(fig, output_path, "score_distribution", save_format)
                if show_plot:
                    fig.show()
                    
            except ImportError:
                print("Warning: Plotly not available. Using matplotlib instead.")
                interactive = False
        
        if not interactive:
            fig, ax, created = self._ensure_fig_ax(ax)
            
            color = palette[0] if palette else 'skyblue'
            ax.hist(y_scores, bins=50, color=color, alpha=0.7, edgecolor='black')
            ax.set_title('Distribution of HALL Scores')
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            
            if save_plot:
                self._save_plot(fig, output_path, "score_distribution", save_format)
            if show_plot and created:
                plt.show()

    # Main plotting interface
    def plot(self, plot_list, metrics_bar=None, show_plots=True, save_plots=False, 
             output_path=None, save_format='png', interactive=False, palette=None, ax=None):
        """
        Plot specified visualizations.
        
        Args:
            plot_list (list): List of plot types to generate
            metrics_bar (list): Metrics to include in bar chart
            show_plots (bool): Whether to display plots
            save_plots (bool): Whether to save plots
            output_path (str): Directory to save plots
            save_format (str): File format for saved plots ('png', 'jpg', 'svg', 'pdf', 'html')
            interactive (bool): Use interactive plotting with plotly (recommended for HTML)
            palette (list): Color palette for plots
            ax: Matplotlib axes object for single plot
            
        Note:
            - For best interactive HTML experience, use interactive=True with save_format='html'
            - Interactive HTML plots include zoom, pan, hover tooltips, and other interactive features
            - Plotly creates more interactive HTML than matplotlib+mpld3
        """
        if save_plots and output_path:
            os.makedirs(output_path, exist_ok=True)
            
        # Recommend interactive mode for HTML format
        if save_format.lower() == 'html' and not interactive:
            print("Tip: For best interactive HTML experience, consider using interactive=True")

        for plot_type in plot_list:
            try:
                if plot_type == 'metrics_bar':
                    metrics_to_plot = metrics_bar or ["f1", "accuracy", "precision", "recall"]
                    self._plot_metrics_bar(metrics_to_plot, show_plots, save_plots, output_path, 
                                         save_format, interactive, palette, ax)
                elif plot_type == 'roc_curve':
                    self._plot_roc_curve(show_plots, save_plots, output_path, save_format, interactive, ax)
                elif plot_type == 'precision_recall_curve':
                    self._plot_precision_recall_curve(show_plots, save_plots, output_path, save_format, interactive, ax)
                elif plot_type == 'confusion_matrix':
                    self._plot_confusion_matrix(show_plots, save_plots, output_path, save_format, interactive, ax)
                elif plot_type == 'score_distribution':
                    self._plot_score_distribution(show_plots, save_plots, output_path, save_format, 
                                                interactive, palette, ax)
                else:
                    raise ValueError(f"Unsupported plot type: {plot_type}")
            except Exception as e:
                print(f"Error plotting {plot_type}: {e}")

    def combined_plots(self, plot_list, title=None, show_plot=True, save_path=None, 
                      save_format='png', figsize=(15, 10), metrics_bar=None):
        """
        Create a combined plot with multiple subplots.
        
        Args:
            plot_list (list): List of plot types to include
            title (str): Overall title for the combined plot
            show_plot (bool): Whether to display the combined plot
            save_path (str): Path to save the combined plot
            save_format (str): File format for saved plot ('png', 'jpg', 'svg', 'pdf', 'html')
            figsize (tuple): Figure size
            metrics_bar (list): Metrics for bar chart
        """
        n_plots = len(plot_list)
        cols = 3
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, plot_type in enumerate(plot_list):
            ax = axes[i]
            try:
                if plot_type == 'metrics_bar':
                    metrics_to_plot = metrics_bar or ["f1", "accuracy", "precision", "recall"]
                    self._plot_metrics_bar(metrics_to_plot, False, False, None, save_format, False, None, ax)
                elif plot_type == 'roc_curve':
                    self._plot_roc_curve(False, False, None, save_format, False, ax)
                elif plot_type == 'precision_recall_curve':
                    self._plot_precision_recall_curve(False, False, None, save_format, False, ax)
                elif plot_type == 'confusion_matrix':
                    self._plot_confusion_matrix(False, False, None, save_format, False, ax)
                elif plot_type == 'score_distribution':
                    self._plot_score_distribution(False, False, None, save_format, False, None, ax)
                else:
                    ax.text(0.5, 0.5, f"Unsupported plot: {plot_type}", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{plot_type} (unsupported)")
            except Exception as e:
                ax.text(0.5, 0.5, f"Error in {plot_type}:\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes, wrap=True)
                ax.set_title(f"{plot_type} (error)")

        # Hide unused subplots
        for j in range(n_plots, len(axes)):
            fig.delaxes(axes[j])

        if title:
            fig.suptitle(title, fontsize=30, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            
            if save_format.lower() == 'html':
                try:
                    import mpld3
                    # Configure mpld3 for better interactivity
                    mpld3.enable_notebook()
                    html_str = mpld3.fig_to_html(fig, template_type='general')
                    with open(save_path, 'w') as f:
                        f.write(html_str)
                    print(f"Interactive combined plot saved to: {save_path}")
                except ImportError:
                    print("Warning: mpld3 not available. Cannot save as interactive HTML.")
                    print("Install with: pip install mpld3")
                    print("For better interactive HTML, consider using individual plots with interactive=True")
                    print("Falling back to PNG format.")
                    save_path = save_path.replace('.html', '.png')
                    fig.savefig(save_path, bbox_inches='tight', dpi=300)
                    print(f"Combined plot saved to: {save_path}")
            else:
                fig.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"Combined plot saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


class myMetric:
    def __init__(self, metric_name: str = None, custom_metric: str = None) -> None:
        return None

    def evaluate_dataset(self) -> None :
        return None

    def save_results(self, folder_path: str, filename: str = "evaluated_dataset", format: str = "hf", **kwargs) -> None:
        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Automatically append the appropriate file extension based on format
        if format == "hf":
            file_path = folder_path  # No filename needed for Hugging Face format
        elif format == "csv":
            file_path = os.path.join(folder_path, f"{filename}.csv")
        elif format == "json":
            file_path = os.path.join(folder_path, f"{filename}.json")
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Save the dataset in the specified format
        if format == "hf":
            self.evaluated_dataset.save_to_disk(folder_path)
            print(f"Dataset saved in Hugging Face format at {folder_path}")
        elif format == "csv":
            self.evaluated_dataset.to_csv(file_path, **kwargs)
            print(f"Dataset saved as CSV at {file_path}")
        elif format == "json":
            self.evaluated_dataset.to_json(file_path, **kwargs)
            print(f"Dataset saved as JSON at {file_path}")


    def benchmark_scores(self, metric_list: list[str] = []):
        if not hasattr(self, "evaluated_dataset"):
            print("no dataset to score")
        scores = scoreManager(dataset=self.evaluated_dataset, metric_list=[])
        return scores



#------------------------------------
# General class for all metrics that "just" use a fine tuned transformer
#  - Include defining model, loading it, providing it to user, running it on dataset
#------------------------------------
class transformer_model(myMetric):
    def __init__(self, model_path: str, tokenizer_path: str, metric_name: str = "custom", custom_metric: bool = True) -> None:
        super().__init__(metric_name=metric_name, custom_metric=custom_metric)

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    #------------------------------------
    # Getter, Loader, Savers
    #------------------------------------
    def load_model(self, save_folder: str) -> None:
        self.model = PreTrainedModel.from_pretrained(self.model_path)


    def load_tokenizer(self, save_folder: str) -> None:
        self.tokenizer.save_pretrained(save_folder)


    def get_model(self) -> PreTrainedModel:
        if not hasattr(self, "model"):
            self.pipeline = self.load_model()
        return self.model


    def get_tokenizer(self) -> PreTrainedTokenizer:
        if not hasattr(self, "model"):
            self.pipeline = self.load_tokenizer()
        return self.tokenizer


    def save_tokenizer(self, save_folder: str) -> None:
        if not hasattr(self, "model"):
            self.pipeline = self.load_tokenizer()

        self.tokenizer.save_pretrained(save_folder)


    def save_model(self, save_folder: str) -> None:
        if not hasattr(self, "model"):
            self.model = self.load_model()

        self.model.save_pretrained(save_folder)


    #------------------------------------
    # Pipeline functions
    #------------------------------------

    #creating pipeline, not done at init to save on space if user only want to save locally or use his own pipeline
    def create_pipeline(self, **kwargs) -> pipeline :
       self.pipeline = pipeline(model=self.model_path, tokenizer=self.tokenizer_path, **kwargs)
       return self.pipeline

    #Execution of the pipeline at row (or batch) level
    #NOTE: Possibly overloaded in metric-specific class
    def run_pipeline(self, batch, source_col, gen_col, top_k, function_to_apply, truncation, padding):
        inputs = [[[source, gen]] for source, gen in zip(batch[source_col], batch[gen_col])]

        results = self.pipeline(inputs, top_k=top_k, truncation=truncation, padding=padding, function_to_apply=function_to_apply)
        return {"predictions": results}

    #Execution of the pipeline on whole dataset
    #NOTE: NOT overloaded in metric-specific class (ideally)
    def evaluate_dataset(self,
        dataset,
        source_col="text",
        gen_col="gen",
        top_k=None,
        truncation=False,
        padding=False,
        function_to_apply=None,
        save_result_dataset_folder_path=None,
        save_result_dataset_format="hf",
        map_kwargs=None
    ):
        dataset = load_to_hf_dataset(dataset)

        if map_kwargs is None:
            map_kwargs = {}

        #if user has not set these parameters inside kwargs then use our default params:
        map_kwargs.setdefault("batched", True)
        map_kwargs.setdefault("batch_size", 10)

        #if not already created, init pipeline
        if not hasattr(self, "pipeline"):
            self.pipeline = self.create_pipeline()

        self.evaluated_dataset = dataset.map(lambda batch: self.run_pipeline(batch=batch,
                                                                             source_col=source_col,
                                                                             gen_col=gen_col,
                                                                             top_k=top_k,
                                                                             truncation=truncation,
                                                                             padding=padding,
                                                                             function_to_apply=function_to_apply), **map_kwargs)

        if save_result_dataset_folder_path:
            self.save_results(folder_path=save_result_dataset_folder_path, format=save_result_dataset_format)

        return self.evaluated_dataset

#------------------------------------
# TrueTeacher
#------------------------------------

class trueTeacher(transformer_model):
    def __init__(self, model_path="google/t5_11b_trueteacher_and_anli", tokenizer_path="google/t5_11b_trueteacher_and_anli") -> None:
        super().__init__(metric_name="trueTeacher",
                         model_path="google/t5_11b_trueteacher_and_anli",
                         tokenizer_path="google/t5_11b_trueteacher_and_anli",
                         custom_metric = False)

    def load_model(self, model_path) -> None:
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    def load_tokenizer(self, tokenizer_path) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

#------------------------------------
# FactCC
#------------------------------------



class factcc(transformer_model):
    def __init__(self, model_path="manueldeprada/FactCC", tokenizer_path="manueldeprada/FactCC") -> None:
        super().__init__(metric_name="factCC", model_path="manueldeprada/FactCC", tokenizer_path="manueldeprada/FactCC", custom_metric = False)


    def convert_label(self, label):
        #We get either 'INCORRECT' or 'CORRECT'
        #'INCORRECT' means there is a hal and 'CORRECT' means there is none
        #   INCORRECT -> HALL
        #   CORRECT   -> NOHALL
        if label == 'INCORRECT':
            return "HALL"
        else:
            return "NOHALL"

    def format_result(self, result):
        #Convert from:
        #   [{'label': 'CORRECT', 'score': 0.9...},
        #    {'label': 'INCORRECT', 'score': 0.1...}]
        #To:
        #   {'CORRECT': 0.9719441533088684, 'INCORRECT': 0.02805584855377674}


        # Check if the result is batched (a list of lists) or not
        if isinstance(result["predictions"][0], list):
            # If batched, iterate over each batch
            predictions = [
                            {self.convert_label(label["label"]): label["score"] for label in batch}
                            for batch in result["predictions"]
                          ]
            score = [ batch["HALL"] for batch in predictions]
            result = {"predictions": predictions,
                      "score" : score}
            return result
        else:
            # If not batched, process directly
            predictions = [
                    {self.convert_label(label["label"]): label["score"] for label in result["predictions"]}
                ]
            score = predictions["HALL"] 
            result = {
                "predictions": predictions,
                "score": score
            }
            return result

    #Execution of the pipeline at row (or batch) level
    # Adding a formatting
    def run_pipeline(self, batch, source_col, gen_col, top_k, function_to_apply, truncation, padding):
        result = super().run_pipeline(batch, source_col, gen_col, top_k, function_to_apply, truncation, padding)
        
        return self.format_result(result)



#------------------------------------
# General class for all metrics that use QG & QA methods
#------------------------------------
class qgqa_based_metric(myMetric):
    def __init__(self,
                 qg_model_path: str,
                 qa_model_path: str,
                 qg_tokenizer_path: str,
                 qa_tokenizer_path: str = None,
                 metric_name="custom",
                 custom_metric=True,
                 qg_prefix="generate questions: ",
                 qg_separator="<sep>"):
        super().__init__(metric_name=metric_name, custom_metric=custom_metric)

        self.qg_model_path = qg_model_path  # Path to the Question Generation model
        self.qa_model_path = qa_model_path  # Path to the Question Answering model
        self.qg_tokenizer_path = qg_tokenizer_path  # Path to the QG Tokenizer
        self.qa_tokenizer_path = qa_tokenizer_path or qg_tokenizer_path  # Use QG Tokenizer if QA Tokenizer not provided
        self.qg_model = None
        self.qa_model = None
        self.qg_tokenizer = None
        self.qa_tokenizer = None
        self.qg_prefix = qg_prefix
        self.qg_separator = qg_separator

#------------------------------------
# Getter, Loader, Savers for QG, QA, and their respective Tokenizers
#------------------------------------

    def load_qg_model(self) -> None:
        self.qg_model = PreTrainedModel.from_pretrained(self.qg_model_path)

    def load_qa_model(self) -> None:
        self.qa_model = PreTrainedModel.from_pretrained(self.qa_model_path)

    def load_qg_tokenizer(self) -> None:
        self.qg_tokenizer = AutoTokenizer.from_pretrained(self.qg_tokenizer_path)

    def load_qa_tokenizer(self) -> None:
        self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_tokenizer_path)

    def get_qg_model(self) -> PreTrainedModel:
        if not self.qg_model:
            self.load_qg_model()
        return self.qg_model

    def get_qa_model(self) -> PreTrainedModel:
        if not self.qa_model:
            self.load_qa_model()
        return self.qa_model

    def get_qg_tokenizer(self) -> AutoTokenizer:
        if not self.qg_tokenizer:
            self.load_qg_tokenizer()
        return self.qg_tokenizer

    def get_qa_tokenizer(self) -> AutoTokenizer:
        if not self.qa_tokenizer:
            self.load_qa_tokenizer()
        return self.qa_tokenizer

    def save_qg_model(self, save_folder: str) -> None:
        if not self.qg_model:
            self.load_qg_model()
        self.qg_model.save_pretrained(save_folder)

    def save_qa_model(self, save_folder: str) -> None:
        if not self.qa_model:
            self.load_qa_model()
        self.qa_model.save_pretrained(save_folder)

    def save_qg_tokenizer(self, save_folder: str) -> None:
        if not self.qg_tokenizer:
            self.load_qg_tokenizer()
        self.qg_tokenizer.save_pretrained(save_folder)

    def save_qa_tokenizer(self, save_folder: str) -> None:
        if not self.qa_tokenizer:
            self.load_qa_tokenizer()
        self.qa_tokenizer.save_pretrained(save_folder)
    
    
    #------------------------------------
    # Pipeline functions
    #------------------------------------

    #creating pipeline, not done at init to save on space if user only want to save locally or use his own pipeline
    def create_qg_pipeline(self, **kwargs) -> pipeline :
        self.qg_pipeline = pipeline(model=self.qg_model_path, tokenizer=self.qg_tokenizer_path, truncation=True, max_length=512, **kwargs)
        return self.qg_pipeline
    
    #creating pipeline, not done at init to save on space if user only want to save locally or use his own pipeline
    def create_qa_pipeline(self, **kwargs) -> pipeline :
       self.qa_pipeline = pipeline(model=self.qa_model_path, tokenizer=self.qa_tokenizer_path, truncation=True, max_length=512, **kwargs)

       return self.qa_pipeline

    #creating pipeline, not done at init to save on space if user only want to save locally or use his own pipeline
    def create_pipeline(self, **kwargs) -> pipeline :
       self.create_qa_pipeline(**kwargs)
       self.create_qg_pipeline(**kwargs)
       return {"qa_pipeline":self.qa_pipeline,"qg_pipeline":self.qg_pipeline}

    # Define the processing function for the question generation input
    # Just adding the prefix
    def format_qg_input(self, batch):
        batch = [self.qg_prefix + text for text in batch]
        return batch

    # Define the processing function for the question generation output
    # (mostly just to deal with the separator and empty questions)
    # Just adding the prefix
    def format_qg_output(self, questions):
        questions = questions.split(self.qg_separator)
        questions = [q.strip() for q in questions if q != ""]

        # Remove duplicates while maintaining order
        seen = set()
        questions = [seen.add(item) or item for item in questions if item not in seen]

        return questions

    #Default function to use to get a score when comparing the answers from the source and the generation
    def compute_token_f1(self, answers, gen_col, src_col):
        f1_results = []

        for answer in answers:
            src_answer = answer[src_col]
            gen_answer = answer[gen_col]

            # Tokenize the answers
            # answers generated with the qa tokenizers so we use this tokenizer here
            tokens_src = set(self.qa_tokenizer.tokenize(src_answer))
            tokens_gen = set(self.qa_tokenizer.tokenize(gen_answer))

            # Calculate precision, recall, and F1 score
            common_tokens = tokens_src & tokens_gen
            precision = len(common_tokens) / len(tokens_src) if tokens_src else 0
            recall = len(common_tokens) / len(tokens_gen) if tokens_gen else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f1_results.append(f1_score)

        return f1_results

    #Getting the score of a single row given the answers generated
    def score_answers(self, answers, gen_col, src_col):
        scores = self.compute_token_f1(answers, gen_col, src_col)

        # Calculate the average score
        if scores:  # Check if the list is not empty
            average_score = sum(scores) / len(scores)
        else:
            average_score = 0
        return average_score

    #Creating and running the pipeline for generating the answer (either from generation or source, passed as context) given the question generated
    def generate_answers(self, question, context):
        answer = self.qa_pipeline(f"question: {question}  context: {context}")[0]["generated_text"]
        return answer

    def generate_questions(self, batch, qg_pipeline_call_args={}):

        #Format to add a prefix ex: "generate question" and only keep the source col
        input = self.format_qg_input(batch)

        #questions = self.qg_pipeline(input, max_new_tokens=max_new_tokens, truncation=truncation)
        questions = self.qg_pipeline(input, **qg_pipeline_call_args)

        #Format the recieved questions ex:
        #[["what does the fox say?<sep>who let the dogs out?<sep>Who you gonna call?"],["Why did the chicken cross the road?<sep>knock knock whos there?"]]
        #[["what does the fox say?","who let the dogs out?","Who you gonna call?"],["Why did the chicken cross the road?","knock knock whos there?"]]
        #the comprehension is for dealing with each line separately, format_qg_output does the formatting and is the one which should be overridden if needed
        questions = [self.format_qg_output(output["generated_text"]) for output in questions]

        return questions

    #Execution of the pipeline at row (or batch) level
    #NOTE: Possibly overloaded in metric-specific class
    def run_pipeline(self, batch, batched, gen_col, src_col, qg_pipeline_call_args, keep_questions, keep_answers):

        # If input is not batched it is not a dict of list
        # Changing it to a list to homogenize rest of code
        # Check if the input is batched (a list of items) or not
        if not batched:
            batch = {key: [value] for key, value in batch.items()}

        #Run the question generation model on input
        src_questions = self.generate_questions(batch[src_col], qg_pipeline_call_args)

        #Run the question answering model on each question
        #Do it in the context of the source, and of the generation
        all_answers_batch = [
            [
                {
                    src_col: self.generate_answers(question, batch[src_col][i]),
                    gen_col: self.generate_answers(question, batch[gen_col][i])
                }
                for question in questions_for_row
            ]
            for i, questions_for_row in enumerate(src_questions)
        ]

        #Give a score to each answer when comparing it between contexts (source/generation) in the same line (default: token-level F1)
        #For each lines, group the scores of all it's answers (default: Average)
        score = [self.score_answers(answers, gen_col, src_col) for answers in all_answers_batch]


        #If it is not batched then I need to send only a str not a list of str
        if not batched:
            score = score[0]

        output={"score": score}

        if keep_questions:
            output["questions"]=src_questions
        if keep_answers:
            output["answers"]=all_answers_batch

        torch.cuda.empty_cache()
        return output

    #Execution of the pipeline on whole dataset
    #NOTE: NOT overloaded in metric-specific class (ideally)
    def evaluate_dataset(self,
        dataset,
        source_col="src",
        gen_col="text",
        keep_questions=False,
        keep_answers=False,
        top_k=None,
        qg_pipeline_call_args={},
        padding=False,
        function_to_apply=None,
        save_result_dataset_folder_path=None,
        save_result_dataset_format="hf",
        map_kwargs=None
    ):
        dataset = load_to_hf_dataset(dataset)

        if map_kwargs is None:
            map_kwargs = {}

        #if user has not set these parameters inside kwargs then use our default params:
        map_kwargs.setdefault("batched", False)
        map_kwargs.setdefault("batch_size", 10)

        #if not already created, init qg & qa pipelines
        if not hasattr(self, "qg_pipeline"):
            self.pipeline = self.create_qg_pipeline()

        if not hasattr(self, "qa_pipeline"):
            self.pipeline = self.create_qa_pipeline()

        #we will need the qa tokenizer to compute differences between answer for the source and the generation
        #we check that it is not loaded and if not we load
        if not self.qa_tokenizer:
            self.load_qa_tokenizer()

        
        with torch.no_grad():
            self.evaluated_dataset = dataset.map(lambda batch: self.run_pipeline(batch=batch,
                                                                             src_col=source_col,
                                                                             gen_col=gen_col,
                                                                             qg_pipeline_call_args=qg_pipeline_call_args,
                                                                             batched=map_kwargs["batched"],
                                                                             keep_questions=keep_questions,
                                                                             keep_answers=keep_answers,
                                                                             ), **map_kwargs)

        torch.cuda.empty_cache()
        
        if save_result_dataset_folder_path:
            self.save_results(folder_path=save_result_dataset_folder_path, format=save_result_dataset_format)

        return self.evaluated_dataset


#------------------------------------
# QAGS
#------------------------------------
#Reproduces the method highlited in QAGS but with other models

class qags(qgqa_based_metric):
    def __init__(self) -> None:
        super().__init__(qg_model_path="valhalla/t5-small-e2e-qg",
                        qa_model_path= "valhalla/t5-small-qa-qg-hl",
                        qg_tokenizer_path= "valhalla/t5-small-e2e-qg",
                        qa_tokenizer_path= "valhalla/t5-small-qa-qg-hl",
                        metric_name="qags",
                        custom_metric=False,
                        qg_prefix="generate questions: ",
                        qg_separator="<sep>")


#------------------------------------
# FEQA
#------------------------------------
#Reproduces the method highlited in FEQA but with other models

class feqa(qgqa_based_metric):
    def __init__(self) -> None:
        super().__init__(qg_model_path="valhalla/t5-base-qg-hl",
                        qa_model_path= "valhalla/t5-small-qa-qg-hl",
                        qg_tokenizer_path= "valhalla/t5-base-qg-hl",
                        qa_tokenizer_path= "valhalla/t5-small-qa-qg-hl",
                        metric_name="feqa",
                        custom_metric=False,
                        qg_prefix="generate questions: ",
                        qg_separator="<sep>")

        self.highlight_token="<hl>"

    #creating pipeline, not done at init to save on space if user only want to save locally or use his own pipeline
    def create_pipeline(self, **kwargs) -> pipeline :
        self.create_qa_pipeline(**kwargs)
        self.create_qg_pipeline(**kwargs)
        try:
            self.spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            # Automatically download the model if it's not found
            from spacy.cli import download
            download("en_core_web_sm")
            self.spacy_model = spacy.load("en_core_web_sm")
        
        return {"qa_pipeline":self.qa_pipeline,"qg_pipeline":self.qg_pipeline, "spacy_model":self.spacy_model}

    #Used to extract all named entities and noun chunks.
    # For each one of them we get a copy of the input where they are enclosed in the defined highlight_token (by default <hl>)
    def spacy_entity_extraction(self, text):
        # Process the text using spaCy
        doc = self.spacy_model(text)

        highlighted_sentences = []

        # Collect all named entities and noun chunks
        mask_targets = list(doc.ents) + list(doc.noun_chunks)

        # Generate a new sentence with each target enclosed in <hl> tags
        for target in mask_targets:
            # Enclose the current target with <hl> tags
            highlighted_text = text.replace(target.text, f"{self.highlight_token}{target.text}{self.highlight_token}")
            highlighted_sentences.append(highlighted_text)

        return highlighted_sentences

    # Define the processing function for the question generation output
    # (mostly just to deal with the separator and empty questions)
    # Just adding the prefix
    def format_qg_output(self, list_of_separated_questions):

        #We get the answser inside the dict
        #and we remove each question beyond the first (we assume it would only be a repetition)
        questions = [ output["generated_text"].split(self.qg_separator)[0] for output in list_of_separated_questions]

        #we remove empty questions
        questions = [q for q in questions if q != ""]

        # Remove duplicates while maintaining order
        seen = set()
        questions = [seen.add(item) or item for item in questions if item not in seen]

        return questions

    def generate_questions(self, batch, qg_pipeline_call_args={}):
        #Creating the masked input for each line
        batch_of_list_of_text_with_entity_hl = [self.spacy_entity_extraction(line) for line in batch]

        #Format to add a prefix ex: "generate question" and only keep the source col
        formatted_batch_of_list_of_text_with_entity_hl = [self.format_qg_input(list_of_text_with_entity_hl) for list_of_text_with_entity_hl in batch_of_list_of_text_with_entity_hl]

        #data in the form:
        #[["generate questions: <hl>entity1.1<hl>...", "generate questions: ...<hl>entity1.2<hl>..."...]",       <- line 1 of input batch
        # ["generate questions: <hl>entity2.1<hl>...", "generate questions: ...<hl>entity2.2<hl>..."...]",       <- line 2 of input batch
        #...]

        #for each line of formatted_batch_of_list_of_text_with_entity_hl we want 1 question per item (entity hl)
        #because we assume there is only a single relevant question per entity hl
        #we generate each question
        all_questions = [self.qg_pipeline(list_of_question, **qg_pipeline_call_args)
                         for list_of_question in formatted_batch_of_list_of_text_with_entity_hl]

        #data in the form:
        #
        #[[{'generated_text': 'question1.1.1<sep>question1.1.2...'},{'generated_text': 'question1.2.1<sep>question1.2.2...'},
        #{'generated_text': 'question2.1.1<sep>question2.1.2...'},{'generated_text': 'question2.2.1<sep>question2.2.2...'},...], <-all for line 1
        #...]
        #
        #   line 1 of input batch, entity 1.X-> y different questions: 1.X.y
        #   line 2 of input batch, entity 2.X-> z different questions: 2.X.z
        #   ...

        #Format the recieved questions ex:
        #[["what does the fox say?<sep>who let the dogs out?<sep>Who you gonna call?"],["Why did the chicken cross the road?<sep>knock knock whos there?"]]
        #[["what does the fox say?","who let the dogs out?","Who you gonna call?"],["Why did the chicken cross the road?","knock knock whos there?"]]
        #the comprehension is for dealing with each line separately, format_qg_output does the formatting and is the one which should be overridden if needed
        questions = [self.format_qg_output(questions_per_line) for questions_per_line in all_questions]

        return questions



#------------------------------------
# General class for all metrics that "just" use a fine tuned transformer
#  - Include defining model, loading it, providing it to user, running it on dataset
#------------------------------------
class entity_based_metric(myMetric):
    def __init__(self, er_model_path, er_tokenizer_path, metric_name="custom", custom_metric=True) -> None:
        super().__init__(metric_name=metric_name, custom_metric=custom_metric)

        self.er_model_path = er_model_path
        self.er_tokenizer_path = er_tokenizer_path

    #------------------------------------
    # Getter, Loader, Savers
    #------------------------------------
    def load_er_model(self, save_folder) -> None:
        self.model = PreTrainedModel.from_pretrained(self.er_model_path)


    def load_er_tokenizer(self, save_folder) -> None:
        self.tokenizer.save_pretrained(save_folder)


    def get_er_model(self) -> PreTrainedModel:
        if not hasattr(self, "model"):
            self.pipeline = self.load_model()
        return self.model


    def get_er_tokenizer(self) -> PreTrainedTokenizer:
        if not hasattr(self, "model"):
            self.pipeline = self.load_tokenizer()
        return self.tokenizer


    def save_er_tokenizer(self, save_folder) -> None:
        if not hasattr(self, "model"):
            self.pipeline = self.load_tokenizer()

        self.tokenizer.save_pretrained(save_folder)


    def save_er_model(self, save_folder) -> None:
        if not hasattr(self, "model"):
            self.model = self.load_model()

        self.model.save_pretrained(save_folder)


    #------------------------------------
    # Pipeline functions
    #------------------------------------

    #creating pipeline, not done at init to save on space if user only want to save locally or use his own pipeline
    def create_pipeline(self, **kwargs) -> pipeline :
       self.create_er_pipeline(**kwargs)
       return {"er_pipeline":self.er_pipeline}

    #creating pipeline, not done at init to save on space if user only want to save locally or use his own pipeline
    def create_er_pipeline(self, **kwargs) -> pipeline :
       self.er_pipeline = pipeline(model=self.er_model_path, tokenizer=self.er_tokenizer_path, truncation=True, max_length=1024, **kwargs)
       return self.er_pipeline

    #Formatting extracted entities
    def format_entities(self, text):
        return text

    def filter_matching_entities(self, entities1, entities2):
        """Filters entities from `entities1` that have matching subject and relation in `entities2`."""
        filtered_entities = [
            entity1 for entity1 in entities1
            if any(entity1["subject"] == entity2["subject"] and entity1["relation"] == entity2["relation"]
                for entity2 in entities2)
        ]
        return filtered_entities

    #Default method to compare entities of both gen and sources to get a score
    def compare_entities(self, src_entities, gen_entities):

        #Filter matching entities based on subject and relation
        gen_entities_prime = self.filter_matching_entities(gen_entities, src_entities)

        #Calculate proporiton of entities and relation in gen that are also present in source
        fact_accuracy = len(gen_entities_prime) / len(gen_entities) if gen_entities else 0
        return fact_accuracy

    #Default method to compare entities of both gen and sources to get a score
    def compare_entities(self, src_entities, gen_entities):

        #Filter matching entities based on subject and relation
        gen_entities_prime = self.filter_matching_entities(gen_entities, src_entities)

        #Calculate proporiton of entities and relation in gen that are also present in source
        fact_accuracy = len(gen_entities_prime) / len(gen_entities) if gen_entities else 0
        return fact_accuracy

    def check_and_replace_none(self, input_data, col_name):
        if isinstance(input_data, list):
            if any(x is None for x in input_data):
                warnings.warn(
                    f"Some entries in column '{col_name}' are None. Converting them to empty strings.",
                    UserWarning
                )
                return [x if x is not None else "" for x in input_data]
        else:
            if input_data is None:
                warnings.warn(
                    f"Input for column '{col_name}' is None. Converting it to an empty string.",
                    UserWarning
                )
                return ""
        return input_data

    
    #Execution of the pipeline at row (or batch) level
    #NOTE: Possibly overloaded in metric-specific class
    def run_pipeline(self, batch, batched, source_col, gen_col, keep_entities, top_k, truncation, padding):

        # If input is not batched it is not a dict of list
        # Changing it to a list to homogenize rest of code
        # Check if the input is batched (a list of items) or not
        if not batched:
            batch = {key: [value] for key, value in batch.items()}

        
        # Check and replace None values in the source and generated columns.
        batch[source_col] = self.check_and_replace_none(batch[source_col], source_col)
        batch[gen_col] = self.check_and_replace_none(batch[gen_col], gen_col)

        
        #Source
        entities_src_tokens = self.er_pipeline(batch[source_col], return_tensors=True, return_text=False)
        entities_src_tokens = [ line["generated_token_ids"] for line in entities_src_tokens]

        entities_src_text = self.er_pipeline.tokenizer.batch_decode(entities_src_tokens)
        entities_src_format = [ self.format_entities(line) for line in entities_src_text]

        #Gen
        entities_gen_tokens = self.er_pipeline(batch[gen_col], return_tensors=True, return_text=False)
        entities_gen_tokens = [ line["generated_token_ids"] for line in entities_gen_tokens]

        entities_gen_text = self.er_pipeline.tokenizer.batch_decode(entities_gen_tokens)
        entities_gen_format = [ self.format_entities(line) for line in entities_gen_text]

        score = [self.compare_entities(src, gen) for src, gen in zip(entities_gen_format, entities_src_format)]

        #If it is not batched then I need to send only a str not a list of str
        if not batched:
            score = score[0]

        output={"score": score}

        if keep_entities:
            if not batched:
                entities_src_format = entities_src_format[0]
                entities_gen_format = entities_gen_format[0]

            output["entities_src"]=entities_src_format
            output["entities_gen"]=entities_gen_format

        return output

    #Execution of the pipeline on whole dataset
    #NOTE: NOT overloaded in metric-specific class (ideally)
    def evaluate_dataset(self,
        dataset,
        source_col="text",
        gen_col="gen",
        top_k=None,
        truncation=False,
        padding=False,
        max_tokens_er = 1000,
        keep_entities=False,
        save_result_dataset_folder_path=None,
        save_result_dataset_format="hf",
        map_kwargs=None
    ):
        dataset = load_to_hf_dataset(dataset)

        if map_kwargs is None:
            map_kwargs = {}
        
        #if user has not set these parameters inside kwargs then use our default params:
        map_kwargs.setdefault("batched", False)
        map_kwargs.setdefault("batch_size", 1)

        #if not already created, init pipeline
        if not hasattr(self, "er_pipeline"):
            self.pipeline = self.create_er_pipeline()

        with torch.no_grad():  # Ensure no gradients
            self.evaluated_dataset = dataset.map(lambda batch: self.run_pipeline(batch=batch,
                                                                                 batched=map_kwargs["batched"],
                                                                                 source_col=source_col,
                                                                                 gen_col=gen_col,
                                                                                 keep_entities=keep_entities,
                                                                                 top_k=top_k,
                                                                                 truncation=truncation,
                                                                                 padding=padding), **map_kwargs)

        if save_result_dataset_folder_path:
            self.save_results(folder_path=save_result_dataset_folder_path, format=save_result_dataset_format)

        return self.evaluated_dataset

#------------------------------------
# Factacc
#------------------------------------

class factacc(entity_based_metric):
    def __init__(self) -> None:
        super().__init__(er_model_path= "Babelscape/rebel-large",
                         er_tokenizer_path= "Babelscape/rebel-large",
                         metric_name="factacc",
                         custom_metric=False)

    #Formatting extracted entities
    def format_entities(self, text):
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'subject': subject.strip(), 'relation': relation.strip(),'object': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'subject': subject.strip(), 'relation': relation.strip(),'object': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'subject': subject.strip(), 'relation': relation.strip(),'object': object_.strip()})
        return triplets


    #Used to compare entities of both gen and sources to get a score
    def compare_entities(self, src_entities, gen_entities):

        #Filter matching entities based on subject and relation
        src_entities_prime = self.filter_matching_entities(src_entities, gen_entities)
        gen_entities_prime = self.filter_matching_entities(gen_entities, src_entities)

        #Calculate the intersection based on matching subject, relation, and object
        intersection = [entity for entity in src_entities_prime
                        if any(entity["subject"] == gen_entity["subject"] and
                            entity["relation"] == gen_entity["relation"] and
                            entity["object"] == gen_entity["object"]
                            for gen_entity in gen_entities_prime)]

        #Calculate factual accuracy as precision
        fact_accuracy = len(intersection) / len(gen_entities_prime) if gen_entities_prime else 0
        return fact_accuracy


