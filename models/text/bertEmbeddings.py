import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertEmbeddings(nn.Module):
    """
    Class for the embedding module. It uses as a pretrained embedding model loaded
    using the transformer library. Also, it includes a tokenizer, this way the module
    recieves input as raw text data and returns the embedded sentence.
    """

    def __init__(self) -> None:
        super(BertEmbeddings, self).__init__()
        """
        Constructor of the BertEmbeddings class.
        """

        # Method to tokenize text
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, inputs) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs: str, the input text.

        Return:
            A tensor, with the embedded sentence, of shape [sequence len, embedding dim]
        """

        tokens = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )
        text_tensor = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))

        # Obtener los embeddings de la oración
        embeddings = self.bert.get_input_embeddings()(text_tensor)

        return embeddings
