class Tokenizer:
    def __init__(
        self,
        keys: list[str],
    ):
        self.keys = keys

    def __call__(self, record: dict) -> list[str]:
        samples = ["<s>"]
        for idx in range(len(record[self.keys[0]])):
            sample = "-".join([f"{record[key][idx]:0.0f}" for key in self.keys])
            samples.append(sample)
        samples.append("</s>")
        return samples


class VelocityTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(keys=["velocity"])
