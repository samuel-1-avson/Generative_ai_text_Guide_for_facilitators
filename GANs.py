import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Generator: takes random noise and generates text sequences
class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, noise):
        embedded = self.embedding(noise)
        output, _ = self.lstm(embedded)
        logits = self.fc(output)
        return logits

# Discriminator: distinguishes real from fake sequences
class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, seq):
        embedded = self.embedding(seq)
        output, _ = self.lstm(embedded)
        validity = torch.sigmoid(self.fc(output[:, -1, :]))  
        return validity

# Hyperparameters
vocab_size = 10000  # Adjust based on your dataset
embedding_dim = 100
hidden_dim = 256
latent_dim = 100
seq_length = 30  # The length of text sequences
batch_size = 64
num_epochs = 50

# Instantiate models
generator = Generator(vocab_size, embedding_dim, hidden_dim)
discriminator = Discriminator(vocab_size, embedding_dim, hidden_dim)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Loss function
adversarial_loss = nn.BCELoss()

# Dummy Dataloader (replace with real data)
# Example: sequences of tokenized text
real_data = torch.randint(0, vocab_size, (batch_size * 10, seq_length))
dataloader = DataLoader(real_data, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for real_seqs in dataloader:
        batch_size = real_seqs.size(0)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real sequences
        real_labels = torch.ones(batch_size, 1)  # Real labels (1s)
        real_preds = discriminator(real_seqs)
        real_loss = adversarial_loss(real_preds, real_labels)

        # Fake sequences
        noise = torch.randint(0, vocab_size, (batch_size, seq_length))  # Random noise
        fake_seqs = generator(noise).argmax(dim=-1)  # Generate fake sequences
        fake_labels = torch.zeros(batch_size, 1)  # Fake labels (0s)
        fake_preds = discriminator(fake_seqs)
        fake_loss = adversarial_loss(fake_preds, fake_labels)

        # Total Discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate fake sequences again
        noise = torch.randint(0, vocab_size, (batch_size, seq_length))
        fake_seqs = generator(noise).argmax(dim=-1)
        fake_preds = discriminator(fake_seqs)

        # We want the fake sequences to be classified as real (1s)
        g_loss = adversarial_loss(fake_preds, real_labels)

        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch}/{num_epochs}, Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}")
