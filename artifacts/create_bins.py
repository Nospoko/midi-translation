import numpy as np
import matplotlib.pyplot as plt
import fortepyan as ff
from fortepyan import MidiPiece
from datasets import load_dataset
import pandas as pd


def find_dataset_dstart_bin_edges(pieces: list[MidiPiece], n_bins: int = 3) -> np.array:

    dstarts = []

    for piece in pieces:
        next_start = piece.df.start.shift(-1)
        dstart = next_start - piece.df.start
        # Last value is nan
        dstarts.append(dstart[:-1])
    # We're not doing num=n_bins + 1 here (like in other functions)
    # Because the last edge is handcraftet ...
    quantiles = np.linspace(0, 1, num=n_bins)

    dstarts = np.hstack(dstarts)
    bin_edges = np.quantile(dstarts, quantiles)[:-1]

    values = pd.DataFrame(np.digitize(dstarts, bin_edges), columns=["value"])

    print(values.head())
    count = pd.pivot_table(values, columns=["value"], aggfunc="size")
    print(count)
    plt.plot(count)
    plt.show()
    bin_edges[0] = 0
    # ... here:
    # dstart is mostly distributed in low values, but
    # we need to have at least one token for longer notes
    last_edge = max(bin_edges[-1] * 3, 0.5)
    bin_edges = np.append(bin_edges, last_edge)
    return bin_edges


def main():

    x = np.linspace(start=0, stop=8, num=99)
    y = (np.exp(x) - 1) / 1000
    plt.plot(y)
    plt.show()
    print(f"  {len(x) + 1}:")
    for val in y:
        print(f"  - {val}")


if __name__ == "__main__":

    dataset = load_dataset("roszcz/maestro-v1", split="train+test+validation")
    pieces = [MidiPiece.from_huggingface(record) for record in dataset]

    y = find_dataset_dstart_bin_edges(pieces=pieces, n_bins=100)
    plt.plot(y)
    plt.show()
    for val in y:
        print(f"  - {val:1.28f}")
