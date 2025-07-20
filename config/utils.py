class DictToObject:
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = DictToObject(value)
            setattr(self, key, value)