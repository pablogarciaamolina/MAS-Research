import torch
from .audio import AlexNet_Based_FCN, Audio_Attention


class Audio_Text_MSA_Model(torch.nn.Module):
    """
    Implementation of multimodal approach on emotion recognition using audio and text.
    Model described in "Audio-and-text-based-emotion-recognition"
    https://github.com/aris-ai/Audio-and-text-based-emotion-recognition/tree/master.

    By our interpretation, that consist on the following modules:

    AUDIO ONLY:
        - Processing of audiofiles contained in de IMEOCAP dataset with conversion to
        spectrogram (used as input data)
        - AlexNet-based FCN module that processes incoming spectrograms
        - Two options:
            - An "attention" layer that processes the outcoming spectrograms and
            obtains the emotion vector
            - A Linear module where the outcoming spectrograms are flattened and
            passed through, obtaining an output vector.

    TEXT ONLY:
        - Processing of text data contained in the IMEOCAP dataset
        - BERT Embedding module that obtains the embedding of every text in a
        high-dimensional vector.

    AUDIO AND TEXT:
        - Concatenation of pair vectors from both audio and text.
        - A Fully Connected (FC) module to obtain the logits of every category
        as the output
    """

    def __init__(
        self,
        f_t_c: tuple[int, int, int],
        num_classes: int,
        seq_dim: int,
        embedding_dim: int,
        out_text_dim: int = 1000,
        C: int = 256,
        lrn_mode: str = "full",
        lambd: float = 0.3,
        dropout: float = 0.4,
    ) -> None:
        """
        Constructor for the model

        Args:
            f_t_c: dimensions of the spectrogram, where f is the in frequency, t
            is the in time, and c in the in channels
            num_classes: number of classes for clasification.
            seq_dim: size of text sequence
            embedding_dim: dimension of text embedding

            out_text_dim: final embedding dimension of text processing
            C: size of channelsin spectrogram representation of the audio. Defaults
            to 256.
            lrn_mode: mode for the LRN in the AlexNet for audio processing, number of
            neighbouring channels to use. Sets to 'full', 'half' or 'single'. Defaults
            to FULL
            lambd: lambda scale factor which controls the uniformity of the
            importance weights of the annotation vectors in the attention layer. Ranges
            between 0 and 1. Defaults to 0.3
            dropout: the probability of dropout for the Dropout layers in FC modules
        """

        super().__init__()

        _, _, c = f_t_c

        # AUDIO ONLY
        self.alexnet = AlexNet_Based_FCN(in_channels=c, C=C, lrn_mode=lrn_mode)
        _, F, T, _ = self.alexnet.get_out_dims(f_t_c)  # type:ignore
        L = F * T
        self.attention = Audio_Attention(L=L, C=C, lambd=lambd)
        self.audio_linear = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=2),
            torch.nn.Linear(L, 1),
        )

        # TEXT ONLY
        self.transform = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(
                in_features=seq_dim * embedding_dim, out_features=out_text_dim
            ),
        )

        # AUDIO AND TEXT
        self.clasificator = torch.nn.Sequential(
            torch.nn.Dropout(dropout), torch.nn.Linear(C + out_text_dim, num_classes)
        )

    def forward(
        self,
        audio_inputs: torch.Tensor,
        text_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward method of the model

        Args:
            audio_inputs: audio spectrogram representation of the audio data.\
            [batch, f, t, c]
            text_inputs: text data representation of the text data. [batch, seq_len]

        Return:
            The logits vector (in tensor form) with size the number of classes.
        """

        # AUDIO ONLY
        out_alexnet = self.alexnet(audio_inputs)
        out_audio = self.attention(out_alexnet)  # [batch, C]
        # out_audio = self.audio_linear(out_alexnet)

        # TEXT ONLY
        out_text = self.transform(text_inputs)

        # AUDIO AND TEXT
        fusioned_features: torch.Tensor = torch.concat(
            [out_audio, out_text], dim=1
        )  # ?
        outputs = self.clasificator(fusioned_features)

        return outputs
