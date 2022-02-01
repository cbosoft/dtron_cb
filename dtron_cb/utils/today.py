from datetime import datetime


def _today() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def tick() -> str:
    global today
    today = _today()
    return today


today = _today()
