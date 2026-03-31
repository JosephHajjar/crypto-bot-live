import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from .dataset import get_dataloaders, TimeSeriesDataset
from .model import LSTMTradingModel, save_model

def train_model(data_path, epochs=50, batch_size=256, learning_rate=1e-3, seq_length=60):
    # Set device to CUDA if available for the 4090 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check for hardware acceleration
    if device.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # 1. Load Data
    print(f"Preparing data loaders...")
    # NOTE: To run this for real, update data_path to actual processed data
    dataset = TimeSeriesDataset(data_path, seq_length=seq_length)
    
    # 80% train, 20% validation
    train_size = int(len(dataset) * 0.8)
    indices = list(range(len(dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Use SequentialSamplers to avoid lookahead bias
    train_sampler = torch.utils.data.SequentialSampler(train_indices)
    val_sampler = torch.utils.data.SequentialSampler(val_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, drop_last=False)
    
    input_dim = len(dataset.feature_cols)

    # 2. Initialize Model
    model = LSTMTradingModel(
        input_dim=input_dim, 
        hidden_dim=128, 
        num_layers=2, 
        output_dim=2, # Binary classification (UP or NOT UP)
        dropout=0.3
    ).to(device)

    # 3. Define Loss Function and Optimizer
    # Look at class imbalance in the training set to weight the loss
    # For trading, predicting UP often has lower frequency than predicting FLAT or DOWN, weighting helps
    # Defaulting to no weights for now
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # Added weight decay for l2 regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    print("Starting training...")
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Add tqdm for visual progress on long runs
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients typical in RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * x_batch.size(0)
            
            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Use no_grad to save memory and compute on validation
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * x_batch.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_model(model, filepath='models/best_model.pth')
            print(f"  --> Saved new best model (Val Loss reduced to {best_val_loss:.4f})")
            
    print(f"Training Complete. Best Validation Loss: {best_val_loss:.4f}, Validation Accuracy: {best_val_acc:.4f}")
    
if __name__ == "__main__":
    # Test invocation wrapper
    import os
    data_path = '../data/data_storage/BTC_USDT_5m_processed.csv'
    if os.path.exists(data_path):
        train_model(data_path, epochs=10, seq_length=60)
    else:
        print(f"Processed data file not found at {data_path}. Generate data first.")
