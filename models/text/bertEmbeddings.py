import torch
import torch.nn as nn
from transformers import BertModel

class BertEmbeddings(nn.Module):
    
    def __init__(self, seq_dim: int, out_dim: int):
        super(BertEmbeddings, self).__init__()
        """
        Constructor of the BertEmbeddings class.

        Args:
            out_dim: output embedding dimension
        """
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        pre_emb_dim: int = self.bert.config.hidden_size

        # Add FC layer
        self.transform = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(in_features=seq_dim*pre_emb_dim, out_features=out_dim),
        )
        
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
        embeddings = self.get_embeddings_(tokens)
        processed_embedding = self.transform(embeddings)
        
        return processed_embedding
    


