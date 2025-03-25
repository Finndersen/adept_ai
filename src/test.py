from inspect import signature

from pydantic_ai._pydantic import function_schema


class Test:
    def method(self, arg1: int, arg2: str | None = None) -> str:
        """
        This is a test method.
        """
        return "Hello, world!"

    def get_schema(self):
        return function_schema(
            function=self.method, takes_ctx=False, docstring_format="auto", require_parameter_descriptions=False
        )


print(Test().get_schema())

print(signature(Test().method))
