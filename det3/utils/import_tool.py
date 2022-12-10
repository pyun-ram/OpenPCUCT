import importlib.util

def load_module(path, name):
    '''
    Note: this function will make the decorator of a function
    works more than one times.
    '''
    spec = importlib.util.spec_from_file_location(name, path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return getattr(foo, name)
