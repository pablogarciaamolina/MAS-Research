# Path: models/pretrainedBERT.py
import torch
import torch.nn as nn
from transformers import BertModel

class PretrainedBERT(nn.Module):
    """
    Pretrained BERT model for fine-tuning.
    
    Args:
        num_classes: int, the number of classes.
    """
    def __init__(self):
        super(PretrainedBERT, self).__init__()
        """
        Constructor of the PretrainedBERT class.
        """
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions = True, output_hidden_states = False)
        self.hidden_size = self.bert.config.hidden_size
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids: torch.Tensor of shape (batch_size, seq_len), the token indices.
            attention_mask: torch.Tensor of shape (batch_size, seq_len), the attention mask.
        """
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs[1]  
        return pooled_output

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        


