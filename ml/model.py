import torch
import torch.nn as nn
import math
import os

class AttentionLSTMModel(nn.Module):
    """
    LSTM with Multi-Head Self-Attention.
    The LSTM captures temporal dependencies, then attention lets the model
    focus on the most relevant timesteps in the sequence for its prediction.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=3, 
                 dropout=0.2, num_heads=4, activation_fn='relu'):
        super(AttentionLSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection to hidden_dim (handles variable input sizes cleanly)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM backbone
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-Head Self-Attention over LSTM outputs
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        
        # Configure activation function
        if activation_fn == 'tanh':
            act = nn.Tanh()
        elif activation_fn == 'leaky_relu':
            act = nn.LeakyReLU()
        else:
            act = nn.ReLU()
            
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # Project input features to hidden_dim
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        
        # Self-attention over all timesteps
        normed = self.layer_norm1(lstm_out)
        attn_out, _ = self.attention(normed, normed, normed)
        
        # Residual connection
        combined = self.layer_norm2(lstm_out + attn_out)
        
        # Use the last timestep for classification
        last = combined[:, -1, :]  # (batch, hidden_dim)
        
        return self.fc(last)

# Keep backward compatibility alias
LSTMTradingModel = AttentionLSTMModel

def save_model(model, filepath='models/best_model.pth'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath='models/best_model.pth', device='cpu'):
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded from {filepath}")
    else:
        print(f"No model found at {filepath}")
    return model

if __name__ == "__main__":
    model = AttentionLSTMModel(input_dim=28, hidden_dim=128, num_layers=2, output_dim=3, activation_fn='leaky_relu')
    dummy_x = torch.randn(32, 60, 28)
    dummy_y = model(dummy_x)
    print(f"Model output shape: {dummy_y.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
