import torch


class Audio_Attention(torch.nn.Module):
    """
    Implementation of the attention module that proceeds the AlexNet-based FCN in the
    model described on "Attention
    Based Fully Convolutional Network for Speech Emotion Recognition"
    (https://arxiv.org/pdf/1806.01506v2.pdf).

    As described, the module recieves a tensor of shape [F, T, C] and transforms
    them into the space A, a L dimensional space of C sized tensors where L = F x T.
    For that, the input tensor will be flattened.

    After that, an MLP layer will be applied to obtain the transformation of A
    requiered.

    Notice, the learnable tensors will be the u and the weights of the MLP. For a
    deeper explanation, see the oficial paper.

    The final result is the tensor c with size C. This vector is the weighted sum of
    the tensors in space A.
    """

    def __init__(self, L: int, C: int, lambd: float = 0.3) -> None:
        """
        This is the constructor for the class

        Args:
            L: size of space A
            C: size of vectors in A
            lambd: lambda scale factor which controls the uniformity of the importance\
                weights of the annotation vectors. Ranges between 0 and 1.\
                Defaults to 0.3
        """

        # call super class constructor
        super().__init__()

        # MLP layer
        # It will recieve a tensor of shape [batch size, C, L] and return a tensor
        # [batch, C, L] with the new representation of A
        # self.mlp = torch.nn.Linear(L, L)

        # Learnable vector u
        self.u = torch.nn.Parameter(torch.randn(C, 1))
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
            The tensor containing the emotion vectors. [batch, C]
        """

        # Reshape the inputs to the space A
        inputs = inputs.permute(0, 3, 1, 2)
        space_A = inputs.view(inputs.shape[0], inputs.shape[1], -1)  # [batch, C, L]

        # # Apply MLP layer to obtain new representations of A
        # space_A = self.mlp(space_A)  # [batch, C, L]

        # Obtain importance weights e [batch, L]
        e = torch.matmul(torch.tanh(space_A).permute(0, 2, 1), self.u).squeeze()

        # Obtain normalized importance weights
        alpha = torch.softmax(self.lambd * e, dim=1)  # [batch, L]

        # Calculate emotion tensor
        c = torch.sum(alpha.unsqueeze(1) * space_A, 2)  # [batch, C]

        return c
