from torch import nn
import torchvision
import torch
from torch.nn import functional as F

# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Load a pre-trained ResNet50 model and remove the last two layers (fully connected and pooling layers)
        resnet = torchvision.models.resnet50(weights=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Add adaptive pooling to output 16x16 spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))

        # Freeze parameters in most layers of ResNet
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Unfreeze parameters in the last layers
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = True

    # Forward pass of the Encoder
    def forward(self, images):
        # Pass the input images through ResNet
        features = self.resnet(images)

        # Reorganize the output into a shape suitable for attention (batch_size, num_pixels, encoder_dim)
        features = features.permute(0, 2, 3, 1)  # Change shape from [batch_size, channels, H, W] to [batch_size, H, W, channels]
        features = features.view(features.size(0), -1, features.size(-1))  # Flatten spatial dimensions
        return features  # Return the processed features
    
# Define the Attention class
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        # Initialize attention dimensions and feature dimensions
        self.attention_dim = attention_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        # Linear layers to project encoder and decoder states to the attention space
        self.encoder_projection = nn.Linear(encoder_dim, attention_dim)
        self.decoder_projection = nn.Linear(decoder_dim, attention_dim)

        # Attention score layer that computes the attention scores
        self.attention_score_layer = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        # Project encoder features to attention space
        encoder_attention = self.encoder_projection(features)  # (batch_size, num_pixels, attention_dim)

        # Project decoder hidden state to attention space
        decoder_attention = self.decoder_projection(hidden_state)  # (batch_size, attention_dim)

        # Combine encoder and decoder projections using tanh activation
        combined_states = torch.tanh(encoder_attention + decoder_attention.unsqueeze(1))  # (batch_size, num_pixels, attention_dim)

        # Compute attention scores for each pixel
        attention_scores = self.attention_score_layer(combined_states)  # (batch_size, num_pixels, 1)

        # Squeeze the last dimension to get attention scores for each pixel
        attention_scores = attention_scores.squeeze(2)  # (batch_size, num_pixels)

        # Apply softmax to normalize attention scores
        alpha = F.softmax(attention_scores, dim=1)  # (batch_size, num_pixels)

        # Compute attention-weighted sum of features
        attention_weights = features * alpha.unsqueeze(2)  # (batch_size, num_pixels, feature_dim)
        attention_weights = attention_weights.sum(dim=1)  # (batch_size, feature_dim)

        return alpha, attention_weights  # Return attention weights and alpha (attention scores)
class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3,device ='cuda:0'):
        super(Decoder,self).__init__()

        # Model parameters
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.device = device

        # Layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def init_hidden_state(self, encoder_out):
        # Compute the mean of the encoder outputs
        mean_encoder_out = encoder_out.mean(dim=1)

        # Initialize hidden and cell states
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)

        return h, c

    def forward(self, features, captions):
        # Step 1: Embed captions
        embeds = self.embedding(captions)

        # Step 2: Initialize LSTM hidden states
        h, c = self.init_hidden_state(features)

        # Step 3: Define sequence parameters
        seq_length = captions.size(1) - 1  # Remove <EOS>
        batch_size = captions.size(0)
        num_features = features.size(1)

        # Step 4: Initialize output tensors
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(self.device)

        # Step 5: Iterate through each sequence step
        for s in range(seq_length):
            # Attention mechanism
            alpha, context = self.attention(features, h)

            # LSTM input: concatenated embedding and context
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            # Output prediction
            output = self.fcn(self.drop(h))
            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas, embeds

class Full_model(nn.Module):
    def __init__(self, embed_dim, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3, device='cuda:0'):
        super(Full_model,self).__init__()

        # Initialize the Encoder and Decoder components
        self.encoder = Encoder()  # The image feature extractor (ResNet-based encoder)

        # Initialize the Decoder which generates the captions from image features and previous tokens
        self.decoder = Decoder(
            embed_size=embed_dim,  # Embedding size of words
            vocab_size=vocab_size,  # Total number of words in vocabulary
            attention_dim=attention_dim,  # Dimension of attention space
            encoder_dim=encoder_dim,  # Feature dimension output by encoder
            decoder_dim=decoder_dim,  # Hidden state dimension of the decoder
            device = device
        )

    def forward(self, images, captions):
        # Extract image features using the encoder
        features = self.encoder(images)

        # Generate predictions (captions), attention scores (alphas), and embeddings (embeds) from the decoder
        preds, alphas, embeds = self.decoder(features, captions)

        return preds, alphas, embeds  # Return predictions, attention weights, and word embeddings


