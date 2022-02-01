import pathlib


def ensure_dir(dir_name: str) -> str:
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    return dir_name
