import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, val_loader, criterion, optimizer, device, max_epochs=100):
    """Train the model with early stopping."""
    early_stopping = EarlyStopping(patience=10)
    best_model = None
    best_val_loss = float('inf')
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                val_loss += loss.item()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(best_model)
    return model

def evaluate_model(model, loader, device):
    """Evaluate model and return predictions, targets, and raw outputs."""
    model.eval()
    predictions = []
    targets = []
    outputs = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(data.y.cpu().numpy())
            outputs.extend(torch.softmax(out, dim=1).cpu().numpy())
    
    return (
        np.array(predictions),
        np.array(targets),
        np.array(outputs)
    )

def compute_metrics(y_true, y_pred):
    """Compute various classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
