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
from transformers import BertForSequenceClassification

class PretrainedBERT(nn.Module):
    """
    Pretrained BERT model for fine-tuning.
    
    Args:
        num_classes: int, the number of classes.
    """
    def __init__(self, num_labels):
        super(PretrainedBERT, self).__init__()
        """
        Constructor of the PretrainedBERT class.
        """
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = num_labels, output_attentions = True, output_hidden_states = False)
        self.hidden_size = self.bert.config.hidden_size
        #self.dropout = nn.Dropout(0.1)
        
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
        return pooled_output

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        


