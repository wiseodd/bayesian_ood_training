from torch import nn


class GRUClassifier(nn.Module):

    def __init__(self, num_classes, vocab_size, feature_extractor=False):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.embedding = nn.Embedding(vocab_size, 50, padding_idx=1)
        self.gru = nn.GRU(input_size=50, hidden_size=128, num_layers=2,
            bias=True, batch_first=True,bidirectional=False)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        hidden = self.features(x)

        if self.feature_extractor:
            return hidden

        logits = self.linear(hidden)
        return logits

    def features(self, x):
        embeds = self.embedding(x)
        hidden = self.gru(embeds)[1][1]  # select h_n, and select the 2nd layer
        return hidden
