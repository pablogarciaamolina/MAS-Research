import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertEmbeddings(nn.Module):
    """
    Class for the embedding module. It uses as a pretrained embedding model loaded
    using the transformer library. Also, it includes a tokenizer, this way the module
    recieves input as raw text data and returns the embedded sentence.
    """

    def __init__(self, sequence_size: int) -> None:
        super(BertEmbeddings, self).__init__()
        """
        Constructor of the BertEmbeddings class.

        Ars:
            sequence_size: sequence dimension. If the final embedding surpasses\
                or does not reach this maxi limit it will be cropped or padded.
        """

        # Method to tokenize text
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.seq_size: int = sequence_size

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

        with torch.no_grad():
            outputs = self.bert(**tokens)
            embeddings = outputs.last_hidden_state

        # Pad sequence
        pad = self.seq_size - embeddings.shape[1]

        if pad > 0:
            out = torch.nn.functional.pad(embeddings, (0, 0, 0, pad), "constant", 0)
        else:
            out = embeddings[:, : self.seq_size, :]

        return out.squeeze()
