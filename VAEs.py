import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Encoder: Encodes text into a latent vector
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, seq):
        embedded = self.embedding(seq)
        _, (h_n, _) = self.lstm(embedded)
        mu = self.fc_mu(h_n[-1]) 
        logvar = self.fc_logvar(h_n[-1])  
        return mu, logvar

# Decoder: Decodes latent vector into text
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, z, seq_length):
        h = self.fc(z).unsqueeze(1).repeat(1, seq_length, 1)
        output, _ = self.lstm(h)
        logits = self.fc_out(output)
        return logits

# Reparameterization Trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# Loss function
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.CrossEntropyLoss()(recon_x.view(-1, vocab_size), x.view(-1))
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

# Hyperparameters
vocab_size = 10000  
embedding_dim = 100
hidden_dim = 256
latent_dim = 50
seq_length = 30
batch_size = 64
num_epochs = 50

# Instantiate models
encoder = Encoder(vocab_size, embedding_dim, hidden_dim, latent_dim)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim, latent_dim)

# Optimizer
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Dummy Dataloader (replace with real data)
real_data = torch.randint(0, vocab_size, (batch_size * 10, seq_length))
dataloader = DataLoader(real_data, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        batch_size = batch.size(0)
        
        # Forward pass through the Encoder
        mu, logvar = encoder(batch)
        z = reparameterize(mu, logvar)

        # Decode the latent vector
        recon_seqs = decoder(z, seq_length)

        # Compute loss
        loss = loss_function(recon_seqs, batch, mu, logvar)
        
        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")
