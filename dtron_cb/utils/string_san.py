def path_safe(s: str) -> str:
    s = s.replace('/', '-')
    return s
