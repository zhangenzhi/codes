import torch
from torch import nn
from transformers import BertConfig, BertModel

class BERTClassifier(nn.Module):
    def __init__(self, seq_length, num_classes, hidden_size=768, num_layers=12, num_attention_heads=12):
        """
        Custom BERT model for classification without pretraining weights.
        Args:
            seq_length (int): The input sequence length.
            num_classes (int): The number of output classes.
            hidden_size (int): The hidden size of the BERT model.
            num_layers (int): Number of encoder layers.
            num_attention_heads (int): Number of attention heads.
        """
        super(BERTClassifier, self).__init__()

        # Define BERT configuration
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,  # Default intermediate size
            max_position_embeddings=seq_length + 2,  # Account for [CLS] and [SEP]
            vocab_size=30522,  # Vocabulary size; can be arbitrary since no pretraining
            type_vocab_size=2,  # For token type embeddings (optional)
        )

        # Define BERT model
        self.bert = BertModel(config)

        # Define classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass for classification.
        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_length).
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_length).
            token_type_ids (torch.Tensor, optional): Token type IDs of shape (batch_size, seq_length).
        Returns:
            torch.Tensor: Logits for each class of shape (batch_size, num_classes).
        """
        # import pdb
        # pdb.set_trace()
        
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

        # Pass through classifier
        logits = self.classifier(cls_output)  # Shape: (batch_size, num_classes)

        return logits
