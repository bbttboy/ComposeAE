# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for text data."""
import string
import numpy as np
import torch


class SimpleVocab(object):

    def __init__(self):
        super(SimpleVocab, self).__init__()
        self.word2id = {}
        self.wordcount = {}
        self.word2id['<UNK>'] = 0
        self.wordcount['<UNK>'] = 9e9

    def tokenize_text(self, text):
        text = text.encode('ascii', 'ignore').decode('ascii')
        tokens = str(text).lower().translate(str.maketrans('', '', string.punctuation)).strip().split()
        return tokens

    # 将文本中的单词添加到单词本，并更新单词id和相应单词计数
    def add_text_to_vocab(self, text):
        tokens = self.tokenize_text(text)
        for token in tokens:
            if token not in self.word2id:
                self.word2id[token] = len(self.word2id)
                self.wordcount[token] = 0
            self.wordcount[token] += 1

    def threshold_rare_words(self, wordcount_threshold=5):
        for w in self.word2id:
            if self.wordcount[w] < wordcount_threshold:
                self.word2id[w] = 0

    # 通过单词本对text进行编码
    def encode_text(self, text):
        tokens = self.tokenize_text(text)
        x = [self.word2id.get(t, 0) for t in tokens]
        return x

    def get_size(self):
        return len(self.word2id)


class TextLSTMModel(torch.nn.Module):

    def __init__(self,
                 texts_to_build_vocab,
                 word_embed_dim=512,
                 lstm_hidden_dim=512):

        super(TextLSTMModel, self).__init__()

        # 循环texts中的text，将其添加到单词本中
        # 即初始化(创建)单词本
        self.vocab = SimpleVocab()
        for text in texts_to_build_vocab:
            self.vocab.add_text_to_vocab(text)
        vocab_size = self.vocab.get_size()

        self.word_embed_dim = word_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        # 输入参数为单词本的词数和需要embedding的维度
        # 网络创建后，输入对应单词id，输出设置维度的向量
        self.embedding_layer = torch.nn.Embedding(vocab_size, word_embed_dim)
        # 参数为'输入维度'和'隐藏层维度'
        self.lstm = torch.nn.LSTM(word_embed_dim, lstm_hidden_dim)
        self.fc_output = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            # Linear是全连接层,参数是‘输入维度’和‘输出维度’
            torch.nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
        )

    def forward(self, x):
        """ input x: list of strings"""
        if type(x) is list:
            if type(x[0]) is str or type(x[0]) is unicode:
                # 通过创建后的单词本 为x中每个句子text编码
                x = [self.vocab.encode_text(text) for text in x]
        # assert-->断言：用于判断一个表达式，在表达式条件为 false 的时候触发异常
        assert type(x) is list
        assert type(x[0]) is list
        # 为什么是int ??!!
        assert type(x[0][0]) is int
        return self.forward_encoded_texts(x)

    # 参数texts是编码后的list，如[2, 6, 8, 1]
    def forward_encoded_texts(self, texts):
        # to tensor
        lengths = [len(t) for t in texts]
        # 加'.long()'是为了使zeros矩阵从浮点类型变成整形
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        # 将texts放入torch.tensor的2维张量中，不足的地方用0占位。
        # 每一列对应每一个编码后的text
        for i in range(len(texts)):
            itexts[:lengths[i], i] = torch.tensor(texts[i])

        # embed words
        # tensor创建时默认cpu类型，加'.cuda()'是将数据类型转换维gpu类型
        # 在这里是将 'torch.LongTensor' 转换为 'torch.cuda.LongTensor'
        # 为什么要将输入放入Variable??!!
        itexts = torch.autograd.Variable(itexts).cuda()
        etexts = self.embedding_layer(itexts)

        # lstm
        lstm_output, _ = self.forward_lstm_(etexts)

        # get last output (using length)
        text_features = []
        for i in range(len(texts)):
            text_features.append(lstm_output[lengths[i] - 1, i, :])

        # output
        text_features = torch.stack(text_features)
        text_features = self.fc_output(text_features)
        return text_features

    def forward_lstm_(self, etexts):
        batch_size = etexts.shape[1]
        first_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_dim),
                        torch.zeros(1, batch_size, self.lstm_hidden_dim))
        first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
        lstm_output, last_hidden = self.lstm(etexts, first_hidden)
        return lstm_output, last_hidden
