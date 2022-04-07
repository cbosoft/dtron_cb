
class License:

    def __init__(self, url: str, idx: int, name: str):
        self.url = url
        self.idx = idx
        self.name = name

    def dict(self) -> dict:
        return dict(url=self.url, id=self.idx, name=self.name)
