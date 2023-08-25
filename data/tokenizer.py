import pandas as pd


class Tokenizer:
    def __init__(
        self,
        keys: list[str],
    ):
        self.keys = keys
        self.specials = ["<s>", "</s>"]

    def __call__(self, record: dict) -> list[str]:
        return self.tokenize(record=record)

    def tokenize(self, record: dict) -> list[str]:
        samples = ["<s>"]
        for idx in range(len(record[self.keys[0]])):
            sample = "-".join([f"{record[key][idx]:0.0f}" for key in self.keys])
            samples.append(sample)
        samples.append("</s>")
        return samples

    def untokenize(self, tokens: list[str]) -> dict:
        sample = pd.DataFrame(columns=self.keys)

        for token in tokens:
            if token in self.specials:
                continue
            values = token.split("-")
            sample = pd.concat([sample, pd.DataFrame([values], columns=self.keys)], axis="rows")

        # I return dict so that untokenize output is the same type as tokenize input
        return sample.to_dict()


class VelocityTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(keys=["velocity"])

    untokenize = property(doc="(!) you cannot untokenize velocity tokens")
