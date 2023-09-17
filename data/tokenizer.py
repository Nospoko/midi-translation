import pandas as pd


class Tokenizer:
    def __init__(
        self,
        keys: list[str],
    ):
        self.keys = keys
        self.specials = ["<s>", "</s>"]

    def __rich_repr__(self):
        yield "Tokenizer"
        yield "keys", self.keys

    def __call__(self, record: dict) -> list[str]:
        return self.tokenize(record=record)

    def tokenize(self, record: dict) -> list[str]:
        samples = ["<s>"]
        for idx in range(len(record[self.keys[0]])):
            sample = "-".join([f"{record[key][idx]:0.0f}" for key in self.keys])
            samples.append(sample)
        samples.append("</s>")
        return samples

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        sample = pd.DataFrame(columns=self.keys)

        for token in tokens:
            if token in self.specials:
                continue

            values_txt = token.split("-")
            values = [eval(txt) for txt in values_txt]

            sample = pd.concat(
                [sample, pd.DataFrame([values], columns=self.keys)],
                axis="rows",
                ignore_index=True,
            )

        return sample


class VelocityTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(keys=["velocity"])

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        values = []
        for token in tokens:
            if token in self.specials:
                continue
            value = eval(token)
            values.append(value)
        sample = pd.DataFrame(values, columns=["velocity"])

        return sample
