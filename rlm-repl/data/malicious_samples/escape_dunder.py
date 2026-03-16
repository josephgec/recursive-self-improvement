# Attempt to escape sandbox via dunder attributes
x = "".__class__.__bases__[0].__subclasses__()
