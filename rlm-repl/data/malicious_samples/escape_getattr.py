# Attempt to use getattr to access forbidden attributes
getattr(__builtins__, '__import__')('os').system('id')
