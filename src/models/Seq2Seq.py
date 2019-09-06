#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import torch.nn as nn
import torch.nn.functional as F
import torch


class Seq2Seq(nn.Module):
    def __init__(self, data_shape):
        super(Seq2Seq, self).__init__()

        self.element_encoder = Encoder(data_shape[2], 15)
        self.offset_encoder = Encoder(data_shape[2], 15)
        self.duration_encoder = Encoder(data_shape[2], 15)

        self.element_decoder = Decoder(45, data_shape[2])
        self.offset_decoder = Decoder(45, data_shape[2])
        self.duration_decoder = Decoder(45, data_shape[2])

    def forward(self, x):
        element_out, element_hn = self.element_encoder(x[:, :, 0, :])
        offset_out, offset_hn = self.offset_encoder(x[:, :, 1, :])
        duration_out, duration_hn = self.duration_encoder(x[:, :, 2, :])

        encodings_out = torch.cat([element_out, offset_out, duration_out], dim=2)
        encodings_h = torch.cat([element_hn, offset_hn, duration_hn], dim=2)

        dec_element_out = self.element_decoder(encodings_out, encodings_h)
        dec_offset_out = self.offset_decoder(encodings_out, encodings_h)
        dec_duration_out = self.duration_decoder(encodings_out, encodings_h)

        dec_element_out = dec_element_out.view(dec_element_out.shape[0], dec_element_out.shape[1], 1, -1)
        dec_offset_out = dec_offset_out.view(dec_element_out.shape[0], dec_element_out.shape[1], 1, -1)
        dec_duration_out = dec_duration_out.view(dec_element_out.shape[0], dec_element_out.shape[1], 1, -1)

        out = torch.cat([dec_element_out, dec_offset_out, dec_duration_out], dim=2)

        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(Encoder, self).__init__()

        self.rnn_forward = nn.GRU(
            input_size=vocab_size,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.rnn_backward = nn.GRU(
            input_size=vocab_size,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h0 = self.get_init_hidden(x)
        fwd, hn_fwd = self.rnn_forward(x, h0)
        h0 = self.get_init_hidden(x)
        bwd, hn_bwd = self.rnn_forward(flip(x, 1), h0)

        return fwd+bwd, hn_fwd+hn_bwd

    def get_init_hidden(self, x):
        """
        Initializes the internal states, depending on the RNN-type.

        :param x: Example input image to get the dimensions
        :return: Initialized internal states
        """
        return torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=input_dim,
            batch_first=True
        )

        self.out = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, state):
        out, _ = self.rnn(F.relu(x), state)
        out = self.out(out)
        return self.softmax(out)

    def get_init_hidden(self, x):
        """
        Initializes the internal states, depending on the RNN-type.

        :param x: Example input image to get the dimensions
        :return: Initialized internal states
        """
        return torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)


def flip(x, dim):
    """
    Flips a dimension (reverse order). BidiLSTM for example uses this feature
    to apply the a LSTM with reversed time step (opposite direction).

    :param x: Tensor, which has a dimension to be flipped. The dimensions of x
        can be arbitrary.
    :param dim: The dimension/axis to be flipped.
    :return: New tensor with flipped dimension

    :example:
        >>> flip([[1,2,3], [4,5,6], [7,8,9]], 0)
        [[7,8,9], [4,5,6], [1,2,3]]
    """
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i) - 1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    return x[inds]

