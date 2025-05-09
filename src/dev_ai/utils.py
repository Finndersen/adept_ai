import functools
import inspect


class InstanceCache:
    """
    Cache decorator for instance methods that take no arguments.
    """

    def __init__(self, func):
        self.func = func
        self.cache_attr = f"_cached_{func.__name__}"

    def __get__(self, instance, owner):
        if instance is None:
            return self

        if inspect.iscoroutinefunction(self.func):

            @functools.wraps(self.func)
            async def async_wrapper():
                if not hasattr(instance, self.cache_attr):
                    setattr(instance, self.cache_attr, await self.func(instance))
                return getattr(instance, self.cache_attr)

            def clear_cache():
                if hasattr(instance, self.cache_attr):
                    delattr(instance, self.cache_attr)

            async_wrapper.clear_cache = clear_cache
            return async_wrapper
        else:

            @functools.wraps(self.func)
            def sync_wrapper():
                if not hasattr(instance, self.cache_attr):
                    setattr(instance, self.cache_attr, self.func(instance))
                return getattr(instance, self.cache_attr)

            def clear_cache():
                if hasattr(instance, self.cache_attr):
                    delattr(instance, self.cache_attr)

            sync_wrapper.clear_cache = clear_cache
            return sync_wrapper


# Decorator form
def cached_method(func):
    return InstanceCache(func)
