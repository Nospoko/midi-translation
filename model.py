import copy

import torch
import torch.nn as nn

from modules.embeddings import Embeddings
from modules.fnn import PositionwiseFeedForward
from modules.decoder import Decoder, DecoderLayer
from modules.encoder import Encoder, EncoderLayer
from modules.attention import MultiHeadedAttention
from modules.positional_encoding import PositionalEncoding
from modules.encoderdecoder import Generator, EncoderDecoder, subsequent_mask


def make_model(
    input_size: int,
    output_size: int,
    n: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> nn.Module:
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h=h, d_model=d_model)
    ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    position = PositionalEncoding(d_model=d_model, dropout=dropout)
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(size=d_model, self_attn=c(attn), feed_forward=c(ff), dropout=dropout), n=n),
        decoder=Decoder(
            DecoderLayer(size=d_model, self_attn=c(attn), src_attn=c(attn), feed_forward=c(ff), dropout=dropout), n=n
        ),
        src_embed=nn.Sequential(Embeddings(d_model=d_model, vocab_size=input_size), c(position)),
        tgt_embed=nn.Sequential(Embeddings(d_model=d_model, vocab_size=output_size), c(position)),
        generator=Generator(d_model, output_size),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1)

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


if __name__ == "__main__":
    run_tests()
