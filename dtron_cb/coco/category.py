
class Category:

    def __init__(self, idx: int, name: str):
        self.idx = idx
        self.name = name

    def dict(self) -> dict:
        return dict(
            id=self.idx,
            name=self.name
        )
