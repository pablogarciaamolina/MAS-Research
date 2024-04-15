import torch
import enum


class LRN_MODES(enum.Enum):
    FULL = "full"
    HALF = "half"
    SINGLE = "single"


class AlexNet_Based_FCN(torch.Module):
    """
    Implementation of FCN based on the AlexNet architecture for emotion recognition
    using on 3D spectrogram representation of audio input. Based on the paper "Attention
    Based Fully Convolutional Network for Speech Emotion Recognition"
    (https://arxiv.org/pdf/1806.01506v2.pdf)

    This implementation will be almost identical to the original AlexNet. However, the
    last three fully connected layers will be omitted  in order to leave only the
    convolutional operations. The sequence of operations, from input to output,
    is the following:

    - Input [f x t x c]
    - 11x11 Conv (96), stride 4
    - Local response normalization n=all(96)
    - 3x3 MaxPool, stride 2
    - 5x5 Conv (256), pad 2
    - Local response normalization n=all(256)
    - 3x3 MaxPool, stride 2
    - 3x3 Conv (384), pad 1
    - 3x3 Conv (384), pad 1
    - 3x3 Conv (256), pad 1
    - 3x3 MaxPool, stride 2
    - Output [F x T x C]


    The schema has been described with the structure `kernelxkernel <Operation>
    (out_channels), stride (optional), pad (optional)`. As well as n the original
    AlexNet implementation, after the first two convolutions, Local Response
    Normalization (LRN) is applied. Be aware, after each convolution a ReLU is applied,
    that means, after the normalization in case it is applied. For brevity, it hasn't
    been included in the schema.

    """

    def __init__(self, in_channels: int, lrn_mode: str = "full") -> None:
        """
        Constructor for the AlexNet-Based FCN

        Args:
            in_channels: input channels
            lrn_mode: mode for the LRN, number of neighbouring channels to use. Sets
            to 'full', 'half' or 'single'
        """

        # call super class constructor
        super().__init__()

        self.fcn: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels=96, kernel_size=(11, 11), stride=4
            ),
            torch.nn.LocalResponseNorm(
                size=self._get_lrn_size(in_channels=96, mode=lrn_mode)
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            torch.nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=(5, 5), padding=2
            ),
            torch.nn.LocalResponseNorm(
                size=self._get_lrn_size(in_channels=256, mode=lrn_mode)
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=(3, 3),
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
        )

    def _get_lrn_size(self, in_channels: int, mode: str) -> int:
        """
        Defines the number of channels to use for LRN based on the input channels and
        the policy to follow

        Args:
            in_channels: the input number of channels
            mode: the policy for the LRN

        Return:
            Teh number of neighbouring channels to use in the LRN.
        """

        assert mode.lower() in LRN_MODES

        if mode == LRN_MODES.FULL:
            return in_channels
        elif mode == LRN_MODES.HALF:
            return in_channels // 2
        elif mode == LRN_MODES.SINGLE:
            return 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward method of the neural network. Returns a processed tensor.


        Args:
            inputs: Input tensor.\
                [batch, f (in frequency dim), t (in time dim),in channels]

        Return:
            Output tensor.\
                [batch, F (out frequency dim),\
                    T (out time dim), out channels = 256 as for the AlexNet]
        """

        # Reshape inputs
        inputs = inputs.permute(
            (0, 3, 1, 2)
        )  # [batch size, in channel size, f, t]

        # Pass through AlexNet FCN
        outputs: torch.Tensor = self.fcn(
            inputs
        )  # [batch size, out channel size, F, C]

        # Reshape to desired output
        outputs = outputs.permute(0, 2, 3, 1) # [batch, F, T, C]

        return outputs

class AudioAttention(torch.Module):
    """
    Implementation of the attention module that proceeds the AlexNet-based FCN in the model described on "Attention
    Based Fully Convolutional Network for Speech Emotion Recognition" (https://arxiv.org/pdf/1806.01506v2.pdf).

    As described, the module recieves a tensor of shape [F, T, C] and transforms them into the space A, a C dimensional space of L sized tensors where L = F x T. For that, the input tensor will be flattened.

    After that, an MLP layer will be applied to obtain the transformation of A requiered.

    Notice, the learnable tensors will be the u and the weights of the MLP. For a deeper explanation, see the oficial paper.

    The final result is the tensor c with size L. This vector is the weighted sum of the tensors in space A, which basicaly means a sum of the vectors for each channel taking into account their importance. 
    """

    def __init__(self, L: int, lambd: float) -> None:
        """
        This is the constructor for the class

        Args:
            L: size of the vectors in space A
            lambd: lambda scale factor which controls the uniformity of the importance weights of the annotation vectors. Ranges between 0 and 1
        """
        
        # call super class constructor
        super().__init__()

        # MLP layer
        # It will recieve a tensor of shape [batch size, C, L] and return a tensor [batch, C, L] with the new representation of A 
        self.mlp = torch.nn.Linear(L, L)

        # Learnable vector u
        self.u = torch.nn.Parameter(torch.randn(L, 1))
        torch.nn.init.xavier_normal_(self.u)

        # Save lambda parameter
        assert 0 <= lambd <= 1
        self.lambd = lambd

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward method for the module. It will return the emotion vector c.

        Args:
            inputs: the inputs tensor. [batch size, F, T, C]

        Return:
            The tensor containing the emotion vectors. [batch, L]
        """

        # Reshape the inputs to the space A
        space_A = inputs.view(inputs.shape[0], inputs.shape[-1], -1) # [batch, C, L]

        # Apply MLP layer to obtain new representations of A
        new_A = self.mlp(space_A)  # [batch, C, L]

        # Obtain importance weights e [batch, C]
        e = torch.matmul(
            self.u.t(),
            torch.tanh(new_A).permute(2, 1, 0)
        ).squeeze().permute(1, 0)

        # Obtain normalized importance weights
        alpha = torch.softmax(self.lambd * e) # [batch, C]

        # Calculate emotion tensor
        assert len((alpha * space_A).shape) == 3 ## !!! earse
        c = torch.sum(alpha * space_A, 1)

        return c