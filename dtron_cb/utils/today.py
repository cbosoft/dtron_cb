from datetime import datetime


def today() -> str:
    return _today


def tick() -> str:
    global _today
    _today = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return _today


_today = tick()
