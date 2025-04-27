import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs):
        hidden = torch.tanh(self.W(inputs))
        weights = torch.softmax(self.v(hidden), dim=1)
        weighted_sum = torch.sum(weights * hidden, dim=1)
        return weighted_sum


class HAN(nn.Module):
    def __init__(self, word_embedding_dim, sentence_embedding_dim, vocabulary_size, num_classes):
        super(HAN, self).__init__()
        self.word_embedding = nn.Embedding(vocabulary_size, word_embedding_dim)
        self.word_encoder = nn.LSTM(word_embedding_dim, sentence_embedding_dim, bidirectional=True, batch_first=True)
        self.word_attention = Attention(sentence_embedding_dim * 2)
        self.sentence_encoder = nn.LSTM(sentence_embeddingirectional=True, batch_first=True)
        self.sentence_attention = Attention(sentence_embedding_dim * 2)
        self.fc = nn.Linear(sentence_embedding_dim * 2, num_classes)

    def forward(self, word_input, sentence_input):
        word_embedded = self.word_embedding(word_input)
        word_output, _ = self.word_encoder(word_embedded)
        word_attention_output = self.word_attention(word_output)

        sentence_embedded = self.word_embedding(sentence_input)
        sentence_output, _ = self.sentence_encoder(sentence_embedded)
        sentence_attention_output = self.sentence_attention(sentence_output)

        concatenated_output = torch.cat([word_attention_output, sentence_attention_output], dim=1)
        output = self.fc(concatenated_output)
        return output
