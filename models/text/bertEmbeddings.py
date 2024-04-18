import torch
import torch.nn as nn
from transformers import BertModel

class BertEmbeddings(nn.Module):
    
    def __init__(self):
        super(BertEmbeddings, self).__init__()
        """
        Constructor of the BertEmbeddings class.
        """
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.embed_dim = self.bert.config.hidden_size
        
    """ def get_embeddings(self, text):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert(**tokens)
            embeddings = outputs.last_hidden_state  # Última capa oculta de BERT
        
        return embeddings"""
    
        
    
    def forward(self, tokens):
        """
        Forward pass of the model.
        
        Args:
            text: str, the input text.
        """
        # Obtener los embeddings de la oración
        embeddings = self.bert.get_input_embeddings()(torch.tensor(tokens))
        
        return embeddings
    


