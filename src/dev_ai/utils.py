import functools
import inspect
from typing import Any, Protocol


class CachedSyncMethod(Protocol):
    def clear_cache(self): ...

    def __call__(self) -> Any: ...


class CachedAsyncMethod(Protocol):
    def clear_cache(self): ...

    async def __call__(self) -> Any: ...


class InstanceCache:
    """
    Cache decorator for instance methods that take no arguments.
    """

    def __init__(self, method):
        self.method = method
        self.cache_attr = f"_cached_{method.__name__}"

    def __get__(self, instance, owner):
        if instance is None:
            return self

        def clear_cache():
            if hasattr(instance, self.cache_attr):
                delattr(instance, self.cache_attr)

        if inspect.iscoroutinefunction(self.method):

            @functools.wraps(self.method)
            async def async_wrapper():
                if not hasattr(instance, self.cache_attr):
                    setattr(instance, self.cache_attr, await self.method(instance))
                return getattr(instance, self.cache_attr)

            async_wrapper.clear_cache = clear_cache
            return async_wrapper
        else:

            @functools.wraps(self.method)
            def sync_wrapper():
                if not hasattr(instance, self.cache_attr):
                    setattr(instance, self.cache_attr, self.method(instance))
                return getattr(instance, self.cache_attr)

            sync_wrapper.clear_cache = clear_cache
            return sync_wrapper


# Decorator form
def cached_method(func) -> CachedSyncMethod | CachedAsyncMethod:
    return InstanceCache(func)
