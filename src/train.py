import torch
import torch.nn as nn
import torch.optim as optim
import os
from .model import ConstellationNet
from .dataset import load_and_preprocess_data, get_dataloaders

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, save_dir="models/saved_models", model_name="constellation_model.pt"):
    """
    Trains the ConstellationNet model using BCE loss and Adam optimizer.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        
        for x_batch, y_batch in train_loader:
            y_batch = y_batch.to(torch.float32)
            outputs = model(x_batch)
            
            # Loss and optimization
            loss = criterion(outputs.reshape(-1), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Train Accuracy
            preds = outputs.round().reshape(-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
        epoch_train_acc = 100 * (correct / total)
        train_accuracies.append(epoch_train_acc)
        
        # Validation
        epoch_val_acc = validate_model(model, val_loader)
        val_accuracies.append(epoch_val_acc)
        
        print(f"Epoch: {epoch+1:2d} | Train Acc: {epoch_train_acc:.3f}% | Val Acc: {epoch_val_acc:.3f}%")
        
    # Save the final model state
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel training complete. Model saved to {save_path}")
    
    return train_accuracies, val_accuracies

def validate_model(model, val_loader):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            outputs = model(x_batch)
            preds = outputs.round().reshape(-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
    return 100 * (correct / total)

if __name__ == "__main__":
    # Standard configuration
    EPOCHS = 10
    LR = 1e-3
    BATCH_SIZE = 10
    
    # Load and prepare data
    X_tr, y_tr, X_val, y_val, X_ts, _ = load_and_preprocess_data()
    train_loader, val_loader = get_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=BATCH_SIZE)
    
    # Initialize and train model
    model = ConstellationNet(nin=X_tr.shape[1], nout=1)
    train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
