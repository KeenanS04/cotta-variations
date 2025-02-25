import re
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def extract_corruption_values(file_path):
    """
    Extracts all integer values from tensor blocks for each corruption type,
    separating them into "ACTUAL" and "PREDICTED" lists.

    Args:
        file_path (str): Path to the input text file.

    Returns:
        dict: A dictionary where each key is a corruption type (e.g. "gaussian_noise5")
              and each value is another dictionary with keys "ACTUAL" and "PREDICTED"
              mapping to lists of integer values.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    corruption_data = {}
    current_block_lines = []   
    current_value_type = None  
    current_corruption = None  
    
    start_pattern = re.compile(r"\[.*?\]:\s+(ACTUAL|PREDICTED)\s+([\w\d_]+):")
    
    def process_block(block_text, value_type, corruption):
        tensor_matches = re.findall(r"tensor\(\[([^\]]+)\]", block_text)
        numbers = []
        for match in tensor_matches:
            nums = re.findall(r'\d+', match)
            numbers.extend([int(n) for n in nums])
        if corruption not in corruption_data:
            corruption_data[corruption] = {"ACTUAL": [], "PREDICTED": []}
        corruption_data[corruption][value_type] = numbers

    for line in lines:
        if line.startswith("["):
            if current_block_lines and current_value_type and current_corruption:
                block_text = "".join(current_block_lines)
                process_block(block_text, current_value_type, current_corruption)
                current_block_lines = []
            start_match = start_pattern.search(line)
            if start_match:
                current_value_type, current_corruption = start_match.groups()
                idx = line.find(":")
                if idx != -1:
                    current_block_lines.append(line[idx+1:])
            else:
                current_value_type = None
                current_corruption = None
        else:
            if current_value_type and current_corruption:
                current_block_lines.append(line)
    if current_block_lines and current_value_type and current_corruption:
        block_text = "".join(current_block_lines)
        process_block(block_text, current_value_type, current_corruption)
    
    return corruption_data

def calculate_metrics(corruption_data):
    """
    Calculates accuracy, precision, recall, and F1-score for each corruption type.
    
    Args:
        corruption_data (dict): Dictionary with keys as corruption types and values as
                                dictionaries with keys "ACTUAL" and "PREDICTED".
    
    Returns:
        dict: A dictionary mapping each corruption type to its metrics.
    """
    results = {}
    for corruption, values in corruption_data.items():
        actual = values.get("ACTUAL", [])
        predicted = values.get("PREDICTED", [])
        
        if len(actual) != len(predicted):
            print(f"Warning: For corruption '{corruption}', the number of ACTUAL values "
                  f"({len(actual)}) does not match the number of PREDICTED values ({len(predicted)}).")
            continue
        
        # Compute metrics using scikit-learn
        acc = accuracy_score(actual, predicted)
        # Here, 'weighted' averaging is used to account for class imbalance.
        prec = precision_score(actual, predicted, average='macro', zero_division=0)
        rec = recall_score(actual, predicted, average='macro', zero_division=0)
        f1 = f1_score(actual, predicted, average='macro', zero_division=0)
        
        results[corruption] = {
            "error": 1- acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
    
    return results