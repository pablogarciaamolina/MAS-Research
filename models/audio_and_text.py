import torch
from .audio import AlexNet_Based_FCN, Audio_Attention
from .text import PretrainedBERT

class Audio_Text_MSA_Model(torch.Module):
    """
    Implementation of multimodal approach on emotion recognition using audio and text.
    Model described in "Audio-and-text-based-emotion-recognition" https://github.com/aris-ai/Audio-and-text-based-emotion-recognition/tree/master.

    By our interpretation that consist on the following modules:

    AUDIO ONLY:
        - Processing of audiofiles contained in de IMEOCAP dataset with conversion to spectrogram (used as input data)
        - AlexNet-based FCN module that processes incoming spectrograms
        - Two options:
            - An "attention" layer that processes the outcoming spectrograms and obtains the emotion vector
            - A Linear module where the outcoming spectrograms are flattened and passed through, obtaining an output vector.
    
    TEXT ONLY:
        - Processing of text data contained in the IMEOCAP dataset
        - BERT Embedding module that obtains the embedding of every text in a high-dimensional vector.

    AUDIO AND TEXT:
        - Concatenation of pair vectors from both audio and text.
        - A Fully Connected (FC) module to obtain the logits of every category as the output
    """

    def __init__(
            self, 
            in_audio_channels: int, 
            L: int,
            C: int,
            hidden_size: int,
            num_classes: int,
            lrn_mode: str = "full", 
            lambd: float = 0.3, 
            dropout: float = 0.4
        ) -> None:
        """
        Constructor for the model

        Args:
            in_channels: input channels in the audio spectrograms.
            L: size of the vectors in space A for audio processing.
            C: size of channelsin spectrogram representation of the audio.
            hidden_size: the size of the hidden states.
            num_classes: number of classes for clasification.
            
            lrn_mode: mode for the LRN in the AlexNet for audio processing, number of neighbouring channels to use. Sets
            to 'full', 'half' or 'single'. Defaults to FULL
            lambd: lambda scale factor which controls the uniformity of the importance weights of the annotation vectors in the attention layer. Ranges between 0 and 1. Defaults to 0.3
            dropout: the probability of dropout for the Dropout layers in FC modules
        """

        super().__init__()

        # AUDIO ONLY
        self.alexnet = AlexNet_Based_FCN(in_channels=in_audio_channels, lrn_mode=lrn_mode)
        self.attention = Audio_Attention(L=L, lambd=lambd)
        self.audio_linear = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=2),
            torch.nn.Linear(L, 1),
        )

        # TEXT ONLY
        self.bert = PretrainedBERT(
            hidden_size=hidden_size,
            num_labels=num_classes
        )

        # AUDIO AND TEXT
        self.clasificator = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(C*hidden_size, num_classes)
        )

    def forward(self,
            audio_inputs: torch.Tensor, 
            text_inputs: torch.Tensor,
            attention_mask: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward method of the model

        Args:
            audio_inputs: audio spectrogram representation of the audio data. [batch, f, t, c]
            text_inputs: text data from the IMEOCAP dataset. [batch, seq_len]
            attention_mask: attention mask for the text data. [batch, seq_len]
            

        Return:
            The logits vector (in tensor form) with size the number of classes.
        """

        # AUDIO ONLY
        out_alexnet = self.alexnet(audio_inputs)
        out_audio = self.attention(out_alexnet)
        # out_audio = self.audio_linear(out_alexnet)

        # TEXT ONLY
        out_text = self.bert(text_inputs, attention_mask)

        # AUDIO AND TEXT
        fusioned_features: torch.Tensor = torch.concat([out_audio, out_text], dim=...) # ?
        outputs = self.clasificator(outputs)