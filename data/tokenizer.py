

class Tokenizer:
    def __init__(
            self,
            keys: list[str],
    ):
        self.keys = keys

    def __call__(self, record: dict) -> list[str]:
        samples = []

        for idx in range(record[self.keys[0]].size):
            sample = str()

            for key in self.keys:
                sample += f"{record[key][idx]}-"

            samples.append(sample[:-1])

        return samples


