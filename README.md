# MAS-Research
Multi-Modal Sentiment Analysis Research

For this project, we aim implement a series of models that help us understand the initial aproaches to the Multi-Model sentiment analysis problem, finding the difficulties that it entails and studying the modern solutions that are currently proposed. Our final goal is to document all our findings throuoghout the process, providing our impressions. Note, that even though we will dive into the most controversial issues, we do not try to implement no state-of-the-art solution, but at most judge possible small improvements.

Finally, be aware this research is conditioned by time as well as computational resources, so its important to highlight that it does not intend to be a repository of potentially useful implemetations, but rather a guide for (our) academic enhacement.


## Project Structure

```
|-- MSA-Research
|   |-- data
|   |   |-- IEMOCAP
|   |   |   |-- Audio
|   |   |   |-- Emotion
|   |   |   |-- Not_Sorted_Audio
|   |   |   |-- Not_Sorted_Emotion
|   |   |   |-- Not_Sorted_Text
|   |   |   |-- Text
|   |-- docs
|   |   |-- MSA_Project_Proposal.pdf
|   |   |-- MSA_Project_Report.pdf
|   |-- models
|   |   |-- audio
|   |   |   |-- __init__.py
|   |   |   |-- attention.py
|   |   |   |-- cnn.py
|   |   |-- text
|   |   |   |-- __init__.py
|   |   |   |-- bertEmbedding.py
|   |   |   |-- myBERT.py
|   |   |   |-- pretrainedBERT.py
|   |   |-- audio_and_text.py
|   |   |-- __init__.py
|   |-- src
|   |   |-- audio_and_text
|   |   |   |-- __init__.py
|   |   |   |-- evaluate.py
|   |   |   |-- train.py
|   |   |   |-- train_functions.py
|   |   |-- data
|   |   |   |-- IEMOCAP
|   |   |   |   |-- __init__.py
|   |   |   |   |-- data.py
|   |   |   |   |-- file_management.py
|   |   |   |-- __init__.py
|   |   |-- __init__.py
|   |   |-- utils.py
|   |-- .gitignore
|   |-- README.md
```

## Models

### Audio-and-Text Model

This model only uses audio and text input. It takes the data from the IEMOCAP dataset.

**Audio processing**

Audio input is passed as a 3D spectrogram representation, with shape `[batch size, f, t, c]` where f is the frequency dimension, t the time dimension and c the channel.

Then is processed by a FCN version of AlexNet into a tensor of shape `[batc, F, T, C]` and passed to an Attention layer to obtain a representation of the data as `[batch, C]`

This attention layer can be substitued by a FC layer.

**Text processing**

The text input is passed already tokenized and then embedded using a pretrained BERT model. After that, were a left with a tensor `[batch, sequence size, embedding dim]` which has its last two dimensions squeezed and passed through a FC layer a transoformed into `[batch, text embedding size]`.

**Audio and Text processing**

For featur fusion we simply concatenate both outputs of the individual processing modules, obtaining a tensor shaped `[batch, C + text embedding dim]`. This is then  passed to a FC layer (with dropout) that act as the classifier.


![Audio-and-Text-Model image](./images/Audio-and-Text-Model_image.png)

