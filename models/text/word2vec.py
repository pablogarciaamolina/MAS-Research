import torch
from gensim.models.keyedvectors import load_word2vec_format
from transformers import BertTokenizer
import gdown

EMBEDDING_DATA_PATH: str = "data/Embeddings/GoogleNews-vectors-negative300.bin.gz"


class Word2Vec_Embedding(torch.nn.Module):
    """
    Class for the Word2Vec-type-based embedding module. It serves as a text processing
    module that recieves a raw text input in sentence-type and returns the
    embedded version.

    The embedding is loaded from memory, and in case it is not in the specified
    folder it will be dowloaded from https://drive.usercontent.google.com/download?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download&authuser=0&resourcekey=0-wjGZdNAUop6WykTtMip30g
    """

    def __init__(self, sequence_size: int) -> None:
        """
        Constructor for the class

        Ars:
            sequence_size: sequence dimension. If the final embedding surpasses\
                or does not reach this maxi limit it will be cropped or padded.
        """

        super().__init__()

        url = "https://drive.usercontent.google.com/download?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download&authuser=0&resourcekey=0-wjGZdNAUop6WykTtMip30g"

        gdown.download(url, EMBEDDING_DATA_PATH, quiet=False)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.word2vec = load_word2vec_format(EMBEDDING_DATA_PATH, binary=True)
        self.seq_size: int = sequence_size

    def _words_to_index(self, words: list[str]) -> torch.Tensor:
        """
        This method transforms a set of words into a tensor containing its
        indexes on the embedding vocabulary.

        Args:
            words: list of words.

        Returns:
            Tensor of shape `[sequence size]`
        """
        idx_tensor = torch.tensor(words).apply_(lambda x: self.word2vec.key_to_index[x])

        return idx_tensor

    def _get_seq(self, tokens: list[str]) -> list[torch.Tensor]:
        """
        This method transforms a list of tokens into their embeddings and then
        concatenates them to forma the output tensor.

        Args:
            tokens: list of tokens

        Return:
            A tensor containing the embedding of the tokens. Shape
            `[num tokens, embedding dim]`
        """

        tensors = []
        for x in tokens:
            try:
                tensors.append(torch.tensor(self.word2vec.get_vector(x)))
            except KeyError:
                pass

        return tensors

    def forward(self, inputs: str) -> torch.Tensor:
        """
        Forward method

        Args:
            inputs: sentence to process

        Return:
            Tensor of shape `[sequence size, embedding dim]`
        """

        # Tokenize the sentence
        tokens: list[str] = self.tokenizer.tokenize(inputs)

        seq: list[torch.Tensor] = self._get_seq(tokens)

        embedded_seq = torch.stack(seq, dim=0)

        # Pad sequence
        pad = self.seq_size - embedded_seq.shape[1]

        if pad > 0:
            out = torch.nn.functional.pad(embedded_seq, (0, 0, 0, pad), "constant", 0)
        else:
            out = embedded_seq[:, : self.seq_size, :]

        return out
