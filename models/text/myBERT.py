# Implementation of BERT model in PyTorch.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyBertModel(nn.Module):
    """
    Class for BERT model.
    
    Args:
        vocab_size: int, the size of the vocabulary.
        hidden_size: int, the size of the hidden states.
        num_attention_heads: int, the number of attention heads in each layer.
        num_hidden_layers: int, the number of hidden layers.
        num_labels: int, the number of labels.
        dropout_prob: float, the dropout probability.
    """
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_hidden_layers, num_labels, dropout_prob=0.1):
        """
        Constructor of the MyBertModel class.
        Layers of the model:
            1. Embedding layer.
            2. BERT encoder: multiple BERT layer instances.
            3. Classifier layer: linear layer to map BERT output to label space.
        """
        super(MyBertModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = BertEncoder(hidden_size, num_attention_heads, num_hidden_layers, dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.
        Args:
            input_ids: torch.Tensor of shape (batch_size, seq_len), the token indices.
            attention_mask: torch.Tensor of shape (batch_size, seq_len), the attention mask.
        """
        embedded = self.embedding(input_ids)
        encoded = self.encoder(embedded, attention_mask)
        pooled_output = encoded[:, 0]  # Take the first token's output
        logits = self.classifier(pooled_output)
        return logits

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)


class BertEncoder(nn.Module):
    """ 
    BERT encoder.
    
    Args:
        hidden_size: int, the size of the hidden states.
        num_attention_heads: int, the number of attention heads in each layer.
        num_hidden_layers: int, the number of hidden layers.
        dropout_prob: float, the dropout probability.
    """
    def __init__(self, hidden_size, num_attention_heads, num_hidden_layers, dropout_prob):
        super(BertEncoder, self).__init__()
        """ 
        Constructor of the BertEncoder class.
        Layers of the encoder:
            1. Multiple BERT layer instances.
        """

        self.layers = nn.ModuleList([BertLayer(hidden_size, num_attention_heads, dropout_prob)
                                     for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertLayer(nn.Module):
    """ 
    BERT layer.
    
    Args:
        hidden_size: int, the size of the hidden states.
        num_attention_heads: int, the number of attention heads in each layer.
        dropout_prob: float, the dropout probability.
    """
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(BertLayer, self).__init__()
        """
        Constructor of the BertLayer class.
        Layers of the layer:
            1. Attention layer.
            2. Intermediate layer.
            3. Output layer.
        """
        self.attention = BertAttention(hidden_size, num_attention_heads, dropout_prob)
        self.intermediate = BertIntermediate(hidden_size)
        self.output = BertOutput(hidden_size, dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    """ 
    BERT attention layer.
    
    Self-attention mechanism: computes attention scores for each token in the sequence
    and applies softmax to obtain the attention probabilities.
    
    Args:
        hidden_size: int, the size of the hidden states.
        num_attention_heads: int, the number of attention heads in each layer.
        dropout_prob: float, the dropout probability.
    """
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(BertAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.num_attention_heads * self.attention_head_size,)
        context_layer = context_layer.view(*new_shape)
        return context_layer


class BertIntermediate(nn.Module):
    """ 
    BERT intermediate layer.
    
    Args:
        hidden_size: int, the size of the hidden states.
    """
    def __init__(self, hidden_size):
        """ 
        Constructor of the BertIntermediate class.
        Layers of the intermediate layer:
            1. Linear layer.
            2. Activation function (ReLU).
        """
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """ 
    BERT output layer.
    Applies a linear transformation followed by dropout
    and adds the residual connection (adds the input to the output).
    
    Args:
        hidden_size: int, the size of the hidden states.
        dropout_prob: float, the dropout probability.
    """
    def __init__(self, hidden_size, dropout_prob):
        """ 
        Constructor of the BertOutput class.
        Layers of the output layer:
            1. Linear layer.
            2. Dropout layer.
        """
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor

