import torch
from gensim.models.keyedvectors import load_word2vec_format
from transformers import BertTokenizer
import gdown

EMBEDDING_DATA_PATH: str = "data/Embeddings/GoogleNews-vectors-negative300.bin.gz"

class Word2Vec_Embedding(torch.nn.Module):

    def __init__(self, sequence_size: int) -> None:

        super().__init__()

        url = "https://drive.usercontent.google.com/download?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download&authuser=0&resourcekey=0-wjGZdNAUop6WykTtMip30g"
        gdown.download(url, EMBEDDING_DATA_PATH, quiet=False)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.word2vec = load_word2vec_format(EMBEDDING_DATA_PATH, binary = True)
        self.seq_size: int = sequence_size

    def _words_to_index(self, words: list[str]) -> torch.Tensor:

        idx_tensor = torch.tensor(words).apply_(lambda x: self.word2vec.key_to_index[x])

        return idx_tensor
    
    def _get_seq(self, tokens) -> list[torch.Tensor]:

        tensors = []
        for x in tokens:
            try:
                tensors.append(torch.tensor(self.word2vec.get_vector(x)))
            except:
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
        tokens = self.tokenizer.tokenize(inputs)

        seq: list[torch.Tensor] = self._get_seq(tokens)

        embedded_seq = torch.stack(seq, dim=0)

        ## Pad sequence
        pad = self.seq_size - embedded_seq.shape[0]

        if pad > 0:
            out = torch.nn.functional.pad(
                embedded_seq,
                (0, 0, 0, pad),
                "constant",
                0
            )
        else:
            out = embedded_seq[:self.seq_size, :]

        return out
    


if __name__ == "__main__":

    sentence = "me gusta nadar"

    embd = Word2Vec_Embedding(5)

    print(embd(sentence))