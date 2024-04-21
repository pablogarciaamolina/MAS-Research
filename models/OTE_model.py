import torch
from .image import ResNet50


class OTE_Model(torch.nn.Module):
    """
    Output Transform Encoder Model for Multimodal Sentimen Analysis, described by Zheng
    Yixiao in
    https://github.com/YeexiaoZheng/Multimodal-Sentiment-Analysis?tab=readme-ov-file

    It has the following arquitecture:

    IMAGE ONLY:
        - The image input is a tensor of shape `[batch, in channels, h, w]`
        - It passes through a ResNet50 module and is converted into
        `[batch, image output dim]`

    TEXT ONLY:
        - The text is processed in the dataprocessing and passes already embedded into
        the model. It has shape `[batch, seq, BERT embedding dim]`
        - This is further processed by a module that flattens that input and passes it
        through a FC to obtain something of shape `[batch, text ouput dim]`

    IMAGE AND TEXT:
        - Both image and text outputs are passed into an attention layer represented
        by a Transformer Encoder. This transformer layer recieves as input the
        concatenation of the output of both image and text individual processing
        modules.
        - After that, we are left with a tensor of shape
        `[batch, image output dim + text output dim]`, which is passed through a final
        classifier module composed of a FC layer and a SoftMax layer. Note the SoftMax
        transformation is done when calculating the loss (Cross Entropy Loss).
        - Final output will be a tensor of shape `[batch, num classes]` containing the
        logits for each class.
    """

    def __init__(
        self,
        image_in_channels: int,
        image_out_dim: int,
        text_seq_dim: int,
        text_embedding_dim: int,
        text_out_dim: int,
        classifier_hidden_size: int,
        num_classes: int,
        dropout: float = 0.4,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
        use_small_cnn: bool = False,
        dim_feed_forward: int = 2048,
    ) -> None:
        """
        Class constructor.

        Args:
            image_in_channels: input channels of images.
            image_out_dim: the output dimension of the processed image.
            text_seq_dim: sequence size of the input embedded text.
            text_embedding_dim: embedding size of the input embedded text.
            text_out_dim: the output dimension of the processed text.
            classifier_hidden_size: the size of the hidden layer in the calsification FC.
            num_classes: number of classes for classification.

            dropout: dropout probability for linear layers. Default to 0.4.
            num_heads: the number of heads in the multihead attention of the\
            Transformer Encoder layer.
            attention_dropout: dropout probability specifically for the Transformer\
            Encoder layer
            use_small_cnn: determines whether to use a smaller CNN instead of the\
            expected ResNet-50 for the image processing. Defaults to False.
            dim_feed_forward: dimension of the feedforward network model in the\
            Transformer Encoder layer. Defaults to 2048.
        """

        super().__init__()

        # IMAGE ONLY
        self.use_small_cnn: bool = use_small_cnn
        if not use_small_cnn:
            self.resnet50 = ResNet50(
                in_channels=image_in_channels, out_dim=image_out_dim
            )
        else:
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(
                    image_in_channels, 64, kernel_size=(7, 7), padding=3, stride=2
                ),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(64, 516, 3, padding=1),
                torch.nn.BatchNorm2d(516, dtype=torch.double),
                torch.nn.ReLU(),
                torch.nn.Conv2d(516, 1024, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(1024, dtype=torch.double),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(-3),
                torch.nn.Dropout(0.6),
                torch.nn.Linear(1024, image_out_dim),
            )

        # TEXT ONLY
        self.embedding_transform = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(
                in_features=text_seq_dim * text_embedding_dim, out_features=text_out_dim
            ),
            torch.nn.ReLU(),
        )

        # IMAGE AND TEXT
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=image_out_dim + text_out_dim,
            nhead=num_heads,
            dropout=attention_dropout,
            dim_feedforward=dim_feed_forward,
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(image_out_dim + text_out_dim, classifier_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(classifier_hidden_size, num_classes),
        )

    def forward(
        self, image_inputs: torch.Tensor, text_inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        The forward method for the model

        Args:
            image_inputs: The input image shaped `[batch, in_channels, h, w]`
            text_inputs: The embedded text inputs shaped\
                `[batch, seq_dim, embedding_dim]`
        """

        # IMAGE ONLY

        if not self.use_small_cnn:
            processed_images: torch.Tensor = self.resnet50(
                image_inputs
            )  # [batch, image_out_dim]
        else:
            processed_images: torch.Tensor = self.cnn(
                image_inputs
            )  # [batch, image_out_dim]

        # TEXT ONLY
        processed_text: torch.Tensor = self.embedding_transform(
            text_inputs
        )  # [batch, text_out_dim]

        # IMAGE AND TEXT
        transformer_output: torch.Tensor = self.transformer(
            torch.cat(
                [
                    processed_images.unsqueeze(0),
                    processed_text.unsqueeze(0),
                ],  # transformer requires input shaped [seq, batch, <dim>]
                dim=2,
            )  # [1, batch, image_out_dim + text_out_dim]
        ).squeeze()
        logits: torch.Tensor = self.classifier(transformer_output)  # [batch, classes]

        return logits
