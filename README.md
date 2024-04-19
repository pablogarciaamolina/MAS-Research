# MAS-Research
Multi-Modal Sentiment Analysis Research


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