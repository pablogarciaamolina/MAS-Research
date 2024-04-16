# Pretrained BERT model for fine-tuning

""" 
In this section, we will use the pretrained BERT model for fine-tuning. Two possible BERT models:
 1. transformers.BertModel:
    Basic BERT model: produces contextualized word embeddings for input tokens. 
    Use: BERT embeddings for custom downstream tasks or to fine-tune BERT for a specific task.
 2. transformers.BertForSequenceClassification: 
    Fine-tuned for sequence classification tasks, where the input is a sequence of tokens and 
    the output is a single label indicating the class of the sequence (sentiment analysis).
"""

# Path: models/pretrainedBERT.py
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForSequenceClassification

class PretrainedBERT(nn.Module):
    """
    Pretrained BERT model for fine-tuning.
    
    Args:
        hidden_size: int, the size of the hidden states.
        num_classes: int, the number of classes.
    """
    def __init__(self, hidden_size, num_labels):
        super(PretrainedBERT, self).__init__()
        """
        Constructor of the PretrainedBERT class.
        Layers of the model:
            1. BERT model.
            2. Classifier layer.
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = num_labels, output_attentions = True, output_hidden_states = False)
        #self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids: torch.Tensor of shape (batch_size, seq_len), the token indices.
            attention_mask: torch.Tensor of shape (batch_size, seq_len), the attention mask.
        """
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs[1]  
        #pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
if __name__ == "__main__":
    # Test the PretrainedBERT model
    model = PretrainedBERT(hidden_size=768, num_labels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)   
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

