

def feature_callback(func, default=None):
    if func is None and default is not None:
        return feature_callback(default)

    elif type(func) == str:
        return lambda x: x[func]

    elif hasattr(func, '__call__'):
        return func

    else:
        raise Exception("Cannot create a callback")
