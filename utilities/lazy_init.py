import inspect
from functools import wraps


class LazyInit:
    def __init__(self, cls):
        self.cls = cls
        self.instance = None

    def __get__(self, instance, owner):
        if self.instance is None:
            self.instance = self.cls()
        return self.instance


def maybe_lazy(*lazy_classes):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for lazy_class in lazy_classes:
                instance = lazy_class()
                for name, value in inspect.getmembers(instance):
                    if not name.startswith('__') and (name not in kwargs or kwargs[name] is None):
                        kwargs[name] = value

            return func(*args, **kwargs)
        return wrapper
    return decorator

