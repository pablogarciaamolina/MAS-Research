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
        
    # Función para obtener embeddings de una oración utilizando BERT
    def get_emebeddings(self, text):
        # Tokenizar la oración y convertirla en tensores
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        # Obtener embeddings de los tokens utilizando BERT
        with torch.no_grad():
            outputs = self.bert(**tokens)
            embeddings = outputs.last_hidden_state  # Última capa oculta de BERT
        
        # Retornar los embeddings de los tokens
        return embeddings
    
    def get_embeddings_(self, tokens):
        """
        Get the embeddings of the input text.
        
        Args:
            text: str, the input text.
        """

        embeddings = self.bert.get_input_embeddings()(torch.tensor(tokens))
        
        return embeddings
        
    
    def forward(self, tokens):
        """
        Forward pass of the model.
        
        Args:
            text: str, the input text.
        """
        # Obtener los embeddings de la oración
        embeddings = self.get_embeddings_(tokens)
        
        return embeddings
    

