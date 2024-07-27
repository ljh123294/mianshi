import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_k, num_heads):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k

        # Linear transformations for queries, keys, and values
        self.W_Q = nn.Linear(input_dim, num_heads * d_k)
        self.W_K = nn.Linear(input_dim, num_heads * d_k)
        self.W_V = nn.Linear(input_dim, num_heads * d_k)

        self.W_O = nn.Linear(input_dim, input_dim)

    def forward(self, X):
        # Compute queries, keys, and values
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # Reshape to (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(Q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(K.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(V.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)

        # Compute output
        output = torch.matmul(attn_weights, V)
        # (batch_size, num_heads, seq_len, d_k)
        output.transpose(1, 2)
        output = output.view(output.size(0), -1, self.num_heads * self.d_k)
        output = self.W_O(output)

        return output

# Example usage
input_dim = 512  # Input dimension
d_k = 64         # Dimension of each attention head
num_heads = 8    # Number of attention heads

# Create an instance of MultiheadAttention
attention = MultiheadAttention(input_dim, d_k, num_heads)

# Generate some input data (batch_size, seq_len, input_dim)
batch_size, seq_len = 4, 10
X = torch.randn(batch_size, seq_len, input_dim)

# Apply multihead attention
output = attention(X)

print("Output shape:", output.shape)
