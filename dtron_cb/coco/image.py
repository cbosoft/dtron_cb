
class Image:

    def __init__(self, image_id: int, file_name: str, height: int, width: int, license_idx=0, date_captured='n/a',
                 **kwargs):
        self.idx = image_id
        self.license_idx = license_idx
        self.file_name = file_name
        self.height = height
        self.width = width
        self.date_captured = date_captured
        _ = kwargs  # explicitly do nothing with kwargs

    def dict(self) -> dict:
        return dict(
            id=self.idx,
            license=self.license_idx,
            file_name=self.file_name,
            height=self.height,
            width=self.width,
            date_captured=self.date_captured
        )
