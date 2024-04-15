# Pretrained BERT model for fine-tuning

# In this section, we will use the pretrained BERT model for fine-tuning. We will use the BERT model from the Hugging Face Transformers library. The BERT model is pretrained on the BookCorpus and English Wikipedia datasets. We will fine-tune the BERT model on the IMDB dataset for sentiment analysis.

# The following code snippet shows how to fine-tune the BERT model on the IMDB dataset for sentiment analysis:

# Path: models/pretrainedBERT.py
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForSequenceClassification

class PretrainedBERT(nn.Module):
    """
    Pretrained BERT model for fine-tuning.
    
    Args:
        hidden_size: int, the size of the hidden states.
        num_classes: int, the number of classes.
    """
    def __init__(self, hidden_size, num_classes):
        super(PretrainedBERT, self).__init__()
        """
        Constructor of the PretrainedBERT class.
        Layers of the model:
            1. BERT model.
            2. Classifier layer.
        """
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 4, output_attentions = True, output_hidden_states = False)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids: torch.Tensor of shape (batch_size, seq_len), the token indices.
            attention_mask: torch.Tensor of shape (batch_size, seq_len), the attention mask.
        """
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs[1]  
        logits = self.classifier(pooled_output)
        return logits

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        

