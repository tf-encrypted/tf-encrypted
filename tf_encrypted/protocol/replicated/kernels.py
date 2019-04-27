# TODO benefits of a bass class??
class Kernel:
    def __call__(self, *args):
        pass


kernels = {}


def register(kernel, dtype_tuple):
    op = kernel.op
    if len(dtype_tuple) != op.num_inputs:
        raise TypeError("Kernel type sig must match number of inputs of op")

    try:
        kernels[op.name][dtype_tuple] = kernel
    except KeyError:
        kernels[op.name] = {}
        kernels[op.name][dtype_tuple] = kernel


def dispatch(context, name, *args, **kwargs):
    dtype_list = []
    for arg in args:
        try:
            dtype_list.append(arg.dtype)
        except AttributeError:
            # TODO make sure arg is a Dtypes
            dtype_list.append(arg)

    dtype_tuple = tuple(dtype_list)
    kernel = kernels[name][dtype_tuple]

    op = kernel.op

    if len(op.attrs) > 0:
        return kernel(context, *args, **kwargs)
    else:
        return kernel(context, *args)
