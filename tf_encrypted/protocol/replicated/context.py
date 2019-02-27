from .kernels import register_all


class Context():
    def __init__(self):
        register_all()
