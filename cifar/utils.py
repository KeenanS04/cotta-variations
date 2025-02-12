import torch
import math
import torch.nn as nn

def evaluate_metrics(accuracy: bool, precision: bool, recall: bool, f1: bool,
                        model: nn.Module,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       batch_size: int = 100,
                       device: torch.device = None):
    results = {'accuracy': None, 'precision': None, 'recall': None, 'f1': None}
    if accuracy:
        results['accuracy'] = accuracy(model, x, y, batch_size, device)
    if precision:
        results['precision'] = precision(model, x, y, batch_size, device)
    if recall:
        results['recall'] = recall(model, x, y, batch_size, device)
    if f1:
        results['f1'] = f1_score(model, x, y, batch_size, device)
    return results

def accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]

def precision(model: nn.Module,
              x: torch.Tensor,
              y: torch.Tensor,
              batch_size: int = 100,
              device: torch.device = None):
    if device is None:
        device = x.device
    true_positives = 0
    false_positives = 0
    n_batches = math.ceil(x.shape[0] / batch_size)

    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].to(device)

            output = model(x_curr)
            predictions = output.max(1)[1]

            true_positives += ((predictions == y_curr) & (y_curr == 1)).sum().item()
            false_positives += ((predictions == 1) & (y_curr == 0)).sum().item()

    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    
def recall(model: nn.Module,
           x: torch.Tensor,
           y: torch.Tensor,
           batch_size: int = 100,
           device: torch.device = None):
    if device is None:
        device = x.device
    true_positives = 0
    false_negatives = 0
    n_batches = math.ceil(x.shape[0] / batch_size)

    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].to(device)

            output = model(x_curr)
            predictions = output.max(1)[1]

            true_positives += ((predictions == y_curr) & (y_curr == 1)).sum().item()
            false_negatives += ((predictions == 0) & (y_curr == 1)).sum().item()

    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

def f1_score(model: nn.Module,
             x: torch.Tensor,
             y: torch.Tensor,
             batch_size: int = 100,
             device: torch.device = None):
    if device is None:
        device = x.device

    prec = precision(model, x, y, batch_size, device)
    rec = recall(model, x, y, batch_size, device)

    return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0