import torch

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, strides: int = 1) -> None:
        """
        Constructor of the Residual Block class. It is composed of two branches.

        - First branch (convolutional branch): 3x3Convolution-padding1(optional
        stride) -> batch norm -> ReLU -> 3x3Convolution-padding1(optional stride) ->
        batch norm

        -  Second branch (residual branch): Passes the input forward, the identity
        function. Optionally, if the input channels are not equal to the output
        channels, a 1x1Convolution(optional stride) is applied in this branch,
        so dimensions match.

        Both branches results add up and are passed to the activation function, ReLU.

        Args:
            input_channels: input channels
            output_channels: output channels
            strides: optional stride value applied to 1st convolution in convolutional
            branch and to 1x1Convolution of residual branch, default to 1
        """

        super().__init__()

        self.conv_branch = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=strides
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(out_channels),
        )

        self.conv_1x1: bool = (
            in_channels != out_channels
        )  # Usefull to simplify computations
        if self.conv_1x1:
            self.residual_branch = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=(1, 1), stride=strides
                ),
                torch.nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual_branch = torch.nn.Sequential()

        self.activation = torch.nn.Sequential(torch.nn.ReLU())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the foward pass.

        Args:
            inputs: batch of tensors.
                Dimensions: [batch, in_channels, height, width]

        Returns:
            batch of tensors. Dimensions: [batch, out_channels,
                (height - 1 / stride) + 1, (width - 1 / stride) + 1].
        """

        # TODO

        inputs0 = inputs.clone()
        y1 = self.conv_branch(inputs)
        y2 = self.residual_branch(inputs0)

        return self.activation(y1 + y2)
    

class BottleneckBlock(torch.nn.Module):
    """
    Deeper ResNet architectures implement a block with 3 convolutional layers instead of 2, this are the BottleNeck blocks.

    Defined in "Deep Residual Learning for Image Recognition" (https://arxiv.org/pdf/1512.03385.pdf)
    """

    def __init__(self, in_channels: int, out_channels: int, strides: int = 1) -> None:
        """
        Constructor of the Bottleneck Block class. It is composed of two branches.

        - First branch (convolutional branch): 1x1Convolution(optional
        stride) -> batch norm  -> ReLU -> 3x3Convolution -> batch norm -> ReLU -> 1x1Convolution(4*out_channels) -> batch norm  -> ReLU

        -  Second branch (residual branch): Passes the input forward, a 
        1x1Convolution(4*out_channels)(optional stride) is applied in this branch,
        so dimensions match.

        Both branches results add up and are passed to the activation function, ReLU.

        Args:
            input_channels: input channels
            output_channels: output channels
            strides: optional stride value applied to 1st convolution in convolutional
            branch and to 1x1Convolution of residual branch, default to 1
        """

        super().__init__()

        self.conv_branch = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1), padding=1, stride=strides
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                out_channels, 4*out_channels, kernel_size=(1, 1)),
            torch.nn.BatchNorm2d(4*out_channels),
            torch.nn.ReLU(),
        )


        self.residual_branch = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, 4*out_channels, kernel_size=(1, 1), stride=strides
            ),
            torch.nn.BatchNorm2d(out_channels),
        )

        self.activation = torch.nn.Sequential(torch.nn.ReLU())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the foward pass.

        Args:
            inputs: batch of tensors.
                Dimensions: [batch, in_channels, height, width]

        Returns:
            batch of tensors. Dimensions: [batch, 4*out_channels,
                (height - 1 / stride) + 1, (width - 1 / stride) + 1].
        """

        # TODO

        inputs0 = inputs.clone()
        y1 = self.conv_branch(inputs)
        y2 = self.residual_branch(inputs0)

        return self.activation(y1 + y2)

class ResNet(torch.nn.Module):
    """
    Class implementing the residual neural network architecture described in
    (He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
    recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern
    Recognition)
    """

    def __init__(
        self, in_channels: int, processing_structure: tuple, num_classes: int, bottleneck_blocks: bool = False
    ) -> None:
        """
        Constructor for ResNet class. Following GoogLeNet policy, ResNet sepparates
        it's architecture in three blocks (data ingest, data processing, prediction)

        - Ingest block: 7x7Convolution(64)-padding3-stride2 -> Batch norm -> ReLU ->
        3x3MaxPooling-padding1-stride2

        - Processing block: Has four modules. Each module is composed of residual
        blocks. In the first module, no 1x1 convolution is set in the first residual
        block. In the other three, there is (with strides=2). Residual blocks of the
        same module mantain the same output channels. In the latter 3 modules, the
        output channels double progressively. Finally, the first module has the same
        number of input channels as output channels.

        - Prediction block: A final GlobalAveragePooling followed by a fully connected
        Linear layer with the desired number of clases as output.

        Args:
            in_channels: input channels
            processing_structure: architecture for the processing block. Contains
            output channels and number of residual blocks per module, take the
            structure:
                `(  <module 1 number of residual blocks: int>,
                ( <latter modules starting out dim: int>,
                (<m2 number of res. blocks: int>, <m3 number of res. blocks: int>,
                <m4 number of res. blocks: int>) ) )`
            num_classes: output. Number of classes for clasification
            bottleneck_blocks: defines whether or not to use Bottleneck block instead of regular residual blocks. Usen in deeper ResNet implementations
        """

        super().__init__()

        # Data Ingestion
        self.ingest_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3, 3), padding=1, stride=2),
        )

        # Data Processing
        # Module 1
        m1_structure: int = processing_structure[0]
        m1 = [ResidualBlock(64, 64) for _ in range(m1_structure)]

        # Modules 2-4
        starting_dim: int = processing_structure[1][0]
        latter_modules_sizes: tuple[int] = processing_structure[1][1]

        block_class = ResidualBlock if not bottleneck_blocks else BottleneckBlock
        in_dim = 64
        out_dim = starting_dim
        m24: list[ResidualBlock] = []
        assert len(latter_modules_sizes) == 3
        for module_size in latter_modules_sizes:
            assert module_size >= 1
            m = [block_class(in_dim, out_dim, strides=2)]
            for _ in range(module_size - 1):
                m.append(block_class(out_dim, out_dim))

            m24 += m
            in_dim = out_dim
            out_dim = 2 * out_dim

        self.processing_block: torch.nn.Module = torch.nn.Sequential(*m1, *m24)

        # Prediction
        self.prediction_block: torch.nn.Module = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(-3),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(out_dim // 2, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of logits.
        It is the output of the neural network.

        Args:
            inputs: batch of images.
                Dimensions: [batch, in_channels, height, width].

        Returns:
            batch of logits. Dimensions: [batch, num_classes].
        """

        y = self.ingest_block(inputs)
        y = self.processing_block(y)
        y = self.prediction_block(y)

        return y


class ResNet18(ResNet):
    """
    Class wrapper of the original ResNet architecture: ResNet-18
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__(
            in_channels,
            processing_structure=(2, (128, (2, 2, 2))),
            num_classes=num_classes,
        )

class ResNet50(ResNet):
    """
    Class wrapper of the original ResNet architecture: ResNet-50
    """

    def __init__(self, in_channels: int, out_dim: int):
        """
        Constructor for the ResNet-50 Module

        Args:
            in_channels: input number of channels. This module expects input of shape `[batch, in_channels, h, w]`
            out_dim: output dimension of the ResNet. Ouput of shape `[batch, out_dim]`
        """

        super().__init__(
            in_channels,
            processing_structure=(3, (128, (4, 6, 3))),
            num_classes=out_dim,
            bottleneck_blocks=True
        )