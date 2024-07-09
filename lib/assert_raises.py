from contextlib import contextmanager


@contextmanager
def assert_raises(exception_type: Exception, message: str = None):
    try:
        yield
    except exception_type as e:
        if message is not None:
            assert str(e) == message
    else:
        raise AssertionError(f"{exception_type.__name__} was not raised")
