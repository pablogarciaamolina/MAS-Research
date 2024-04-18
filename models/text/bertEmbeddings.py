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
    def get_embeddings(self, text):
        # Tokenizar la oración y convertirla en tensores
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        # Obtener embeddings de los tokens utilizando BERT
        with torch.no_grad():
            outputs = self.bert(**tokens)
            embeddings = outputs.last_hidden_state  # Última capa oculta de BERT
        
        return embeddings
    
    def get_embeddings_(self, tokens):
        """
        Get the embeddings of the input text.
        
        Args:
            text: str, the input text.
        """

<<<<<<< HEAD
        embeddings = self.bert.get_input_embeddings()(torch.tensor(tokens))
=======
        tokens = self.tokenizer(text)
        token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        embeddings = self.bert.get_input_embeddings()(token_ids)
>>>>>>> a1d98c1e5cba0f1c0655b4955067ef75c375194c
        
        return embeddings
        
    
    def forward(self, tokens):
        """
        Forward pass of the model.
        
        Args:
            text: str, the input text.
        """
        # Obtener los embeddings de la oración
<<<<<<< HEAD
        embeddings = self.get_embeddings_(tokens)
=======
        embeddings = self.get_embeddings(text)
>>>>>>> a1d98c1e5cba0f1c0655b4955067ef75c375194c
        
        return embeddings
    


