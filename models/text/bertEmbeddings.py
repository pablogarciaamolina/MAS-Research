import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertEmbeddings(nn.Module):
    def __init__(self):
        super(BertEmbeddings, self).__init__()
        """
        Constructor of the BertEmbeddings class.
        """

        # Method to tokenize text
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # pre_emb_dim: int = self.bert.config.hidden_size

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs: str, the input text.
        """

        tokens = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )
        text_tensor = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))

        # Obtener los embeddings de la oración
        embeddings = self.bert.get_input_embeddings()(text_tensor)

        return embeddings
