import subprocess
import sys
import importlib.util
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import spacy

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


def load_to_hf_dataset(obj):
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
            return Dataset.from_json(obj)
        elif file_ext in [".csv", ".xls", ".xlsx"]:
            # Load CSV or Excel file into pandas first
            df = pd.read_csv(obj) if file_ext == ".csv" else pd.read_excel(obj)
            return Dataset.from_pandas(df)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    else:
        raise TypeError("Unsupported input type. Expected pandas.DataFrame, Hugging Face Dataset, or file path.")

# Usage examples:
# df = pd.DataFrame(...)  # Example DataFrame
# ds = load_to_hf_dataset(df)

# hf_dataset = Dataset.from_dict({"col1": [1, 2], "col2": ["a", "b"]})
# ds = load_to_hf_dataset(hf_dataset)

# ds = load_to_hf_dataset("data.json")


#------------------------------------
# Class to generate accuracy, recall etc... and graphs on annotated evaluated datasets
#------------------------------------


class ScoreManager:
    def __init__(self, dataset: Dataset, metric_list: list[str], on_split: str ="test"):
        """
        Initialize the ScoreManager with dataset and metrics to compute.

        Args:
            dataset (dict): The dataset containing 'label' and 'predictions' keys.
            metric_list (list): List of metric names to compute. Supported metrics:
                                ['f1', 'precision', 'recall', 'accuracy',
                                 'balanced_accuracy', 'mcc', 'kappa',
                                 'log_loss', 'roc_values', 'auc',
                                 'confusion_matrix', 'precision_recall_values'].
        """

        # Check if the dataset has splits (is a DatasetDict)
        if isinstance(dataset, DatasetDict):
            if on_split is None:
                raise ValueError("The dataset has splits, but no split was provided. Please specify a split (e.g., 'train', 'test').")
            # Select the provided split
            if on_split in dataset:
                dataset = dataset[on_split]
            else:
                raise ValueError(f"Split '{on_split}' does not exist in the dataset. Available splits: {list(dataset.keys())}")
        # If no splits, use the entire dataset dont need to change anything
        display(dataset)

        self.labels = dataset["label"]
        self.class_scores = dataset["predictions"]
        self.predicted_labels = [self.get_predicted_label(example) for example in dataset]
        self.metric_list = metric_list

        # Initialize metrics attributes
        self.results = {}
        self.calculate_metrics()

        # Calculate and store ROC and precision-recall results
        #self.results['roc_auc'], self.results['roc_values'] = self.calculate_roc()
        #self.results['pr_values'] = self.calculate_precision_recall()

        # Calculate confusion matrix
        #self.results['confusion_matrix'] = self.calculate_confusion_matrix()

    def get_predicted_label(self, example):
        """Get the label with the highest score from predictions."""
        prediction_dict = example["predictions"]
        return max(prediction_dict, key=prediction_dict.get)

    def calculate_metrics(self):
        """Calculate and store the specified metrics."""
        for metric in self.metric_list:
            method_name = f"calculate_{metric}"
            if hasattr(self, method_name):
                self.results[metric] = getattr(self, method_name)()
            else:
                raise ValueError(f"Unsupported metric: {metric}")

    def calculate_f1(self):
        """Calculate F1 score."""
        return f1_score(self.labels, self.predicted_labels, average='weighted')

    def calculate_precision(self):
        """Calculate precision score."""
        return precision_score(self.labels, self.predicted_labels, average='weighted')

    def calculate_recall(self):
        """Calculate recall score."""
        return recall_score(self.labels, self.predicted_labels, average='weighted')

    def calculate_accuracy(self):
        """Calculate accuracy score."""
        return accuracy_score(self.labels, self.predicted_labels)

    def calculate_balanced_accuracy(self):
        """Calculate balanced accuracy score."""
        return balanced_accuracy_score(self.labels, self.predicted_labels)

    def calculate_mcc(self):
        """Calculate Matthews Correlation Coefficient."""
        return matthews_corrcoef(self.labels, self.predicted_labels)

    def calculate_kappa(self):
        """Calculate Cohen's Kappa score."""
        return cohen_kappa_score(self.labels, self.predicted_labels)

    def calculate_log_loss(self):
        """Calculate Log Loss (Cross-Entropy Loss)."""
        y_pred_proba = [list(example.values()) for example in self.class_scores]
        return log_loss(self.labels, y_pred_proba)

    def calculate_roc_values(self):
        """Calculate ROC curve and AUC."""
        y_true = np.array([1 if label == 'HALL' else 0 for label in self.labels])
        y_scores = [example['HALL'] for example in self.class_scores]

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

    def calculate_auc(self):
        """Calculate ROC curve and AUC."""
        y_true = np.array([1 if label == 'HALL' else 0 for label in self.labels])
        y_scores = [example['HALL'] for example in self.class_scores]

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        return roc_auc

    def calculate_precision_recall_values(self):
        """Calculate Precision-Recall curve values."""
        y_true = np.array([1 if label == 'HALL' else 0 for label in self.labels])
        y_scores = [example['HALL'] for example in self.class_scores]

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        return {'precision': precision, 'recall': recall, 'thresholds': thresholds}

    def calculate_confusion_matrix(self):
        """Calculate confusion matrix."""
        cm = confusion_matrix(self.labels, self.predicted_labels, labels=['HALL', 'NOHALL'])
        return cm

    def plot_confusion_matrix(self):
        """Plot the confusion matrix."""
        if "confusion_matrix" not in self.results:
            self.results["confusion_matrix"] = self.calculate_confusion_matrix()

        cm = self.results['confusion_matrix']
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(['HALL', 'NOHALL']))
        plt.xticks(tick_marks, ['HALL', 'NOHALL'])
        plt.yticks(tick_marks, ['HALL', 'NOHALL'])

        threshold = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        return plt

    def plot_metrics_bar_chart(self, metrics_to_plot):
        """Plot a bar chart for specified metrics."""
        values = [self.results[metric] for metric in metrics_to_plot if metric in self.results]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_to_plot, values, color=plt.cm.viridis(np.linspace(0, 1, len(values))))

        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Metrics Bar Chart')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)  # Adjust y-axis as necessary

        # Adding value labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

        plt.tight_layout()
        return plt

    def plot_roc_curve(self):
        """Plot ROC curve."""
        if "roc_values" not in self.results:
            self.results["roc_values"] = self.calculate_roc_values()
        if "roc_auc" not in self.results:
            self.results["roc_auc"] = self.calculate_auc()

        roc_values = self.results['roc_values']
        plt.figure()
        plt.plot(roc_values['fpr'], roc_values['tpr'], color='blue', label='ROC curve (area = {:.2f})'.format(self.results['roc_auc']))
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        return plt

    def plot_precision_recall_curve(self):
        """Plot Precision-Recall curve."""
        if "pr_values" not in self.results:
            self.results["roc_values"] = self.calculate_precision_recall_values()

        pr_values = self.results['precision_recall_values']
        plt.figure()
        plt.plot(pr_values['recall'], pr_values['precision'], color='blue')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        return plt

    def plot(self, plot_list, metrics_bar=None, save_plots=False, output_path=None):
        """Plot specified metrics and visualizations from the provided list.

        Args:
            plot_list (list): List of plot types to generate.
            metrics_bar (list, optional): Metrics to plot in the bar chart.
            save_plots (bool, optional): Whether to save the plots to files.
            output_path (str, optional): Path to save plots if save_plots is True.
        """
        for plot_type in plot_list:
            if plot_type == 'metrics_bar' and metrics_bar:
                bar_plot = self.plot_metrics_bar_chart(metrics_bar)
                if save_plots and output_path:
                    bar_plot.savefig(f'{output_path}/metrics_bar_chart.png')
                else:
                    plt.show()
            elif plot_type == 'roc_curve':
                roc_plot = self.plot_roc_curve()
                if save_plots and output_path:
                    roc_plot.savefig(f'{output_path}/roc_curve.png')
                else:
                    plt.show()
            elif plot_type == 'precision_recall_curve':
                pr_plot = self.plot_precision_recall_curve()
                if save_plots and output_path:
                    pr_plot.savefig(f'{output_path}/precision_recall_curve.png')
                else:
                    plt.show()
            elif plot_type == 'confusion_matrix':
                cm_plot = self.plot_confusion_matrix()
                if save_plots and output_path:
                    cm_plot.savefig(f'{output_path}/confusion_matrix.png')
                else:
                    plt.show()
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")



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

        if map_kwargs is None:
            map_kwargs = {}

        #if user has not set these parameters inside kwargs then use our default params:
        map_kwargs.setdefault("batched", False)
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
            return {
                "predictions": [
                    {self.convert_label(label["label"]): label["score"] for label in batch}
                    for batch in result["predictions"]
                ]
            }
        else:
            # If not batched, process directly
            return {
                "predictions": [
                    {self.convert_label(label["label"]): label["score"] for label in result["predictions"]}
                ]
            }

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
        questions = [q for q in questions if q != ""]

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
                                    {src_col: self.generate_answers(question, src_col),
                                    gen_col: self.generate_answers(question, gen_col)}
                                for question in row]
                            for row in src_questions]

        #Give a score to each answer when comparing it between contexts (source/generation) in the same line (default: token-level F1)
        #For each lines, group the scores of all it's answers (default: Average)
        score = [self.score_answers(answers, gen_col, src_col) for answers in all_answers_batch]


        #If it is not batched then I need to send only a str not a list of str
        if not batched:
            score = score[0]

        output={"scores": score}

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

    #Execution of the pipeline at row (or batch) level
    #NOTE: Possibly overloaded in metric-specific class
    def run_pipeline(self, batch, batched, source_col, gen_col, keep_entities, top_k, truncation, padding):

        # If input is not batched it is not a dict of list
        # Changing it to a list to homogenize rest of code
        # Check if the input is batched (a list of items) or not
        if not batched:
            batch = {key: [value] for key, value in batch.items()}

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

        output={"scores": score}

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


