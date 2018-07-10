import tensorflow as tf

from ..protocol import Protocol
from ..tensor.int100 import Int100Tensor as Tensor
from ..tensor.int100 import Int100Placeholder as Placeholder
from ..tensor.int100 import Int100Variable as Variable
from ..tensor.helpers import *

BITPRECISION_INTEGRAL   = 16
BITPRECISION_FRACTIONAL = 16
TRUNCATION_GAP = 20

M = Tensor.modulus
INT_TYPE = Tensor.int_type
K = 2 ** BITPRECISION_FRACTIONAL

assert log2(M) >= 2 * (BITPRECISION_INTEGRAL + BITPRECISION_FRACTIONAL) + log2(1024) + TRUNCATION_GAP

def _encode(rationals, precision=BITPRECISION_FRACTIONAL):
    """ [NumPy] Encode tensor of rational numbers into tensor of ring elements """
    return (rationals * (2**precision)).astype(int).astype(object) % M

def _decode(elements, precision=BITPRECISION_FRACTIONAL):
    """ [NumPy] Decode tensor of ring elements into tensor of rational numbers """
    map_negative_range = np.vectorize(lambda element: element if element <= M/2 else element - M)
    return map_negative_range(elements).astype(float) / (2**precision)

def _share(secret):
    assert isinstance(secret, Tensor), type(secret)

    with tf.name_scope('share'):
        share0 = Tensor.sample_uniform(secret.shape)
        share1 = secret - share0
        return share0, share1

def _reconstruct(share0, share1):
    assert isinstance(share0, Tensor), type(share0)
    assert isinstance(share1, Tensor), type(share1)

    with tf.name_scope('reconstruct'):
        return share0 + share1

def _mul_public_public(prot, x, y):
    assert isinstance(x, [PublicTensor]), type(x)
    assert isinstance(y, [PublicTensor]), type(y)

    x_on_0, x_on_1 = x.unwrapped
    y_on_0, y_on_1 = y.unwrapped

    with tf.name_scope('mul'):

        with tf.device(prot.server_0.device_name):
            z_on_0 = x_on_0 * y_on_0

        with tf.device(prot.server_1.device_name):
            z_on_1 = x_on_1 * y_on_1

    z = PublicTensor(prot, z_on_0, z_on_1)
    


def _mul_public_private(prot, x, y):
    pass

def _mul_private_public(prot, x, y):
    pass

def _mul_private_private(prot, x, y):
    assert isinstance(x, [PrivateTensor, MaskedPrivateTensor]), type(x)
    assert isinstance(y, [PrivateTensor, MaskedPrivateTensor]), type(y)

    node_key = ('mul', x, y)
    z = _nodes.get(node_key, None)

    if z is None:

        if isinstance(x, PrivateTensor):
            x = mask(x)

        if isinstance(y, PrivateTensor):
            y = mask(y)

        assert type(x) in [MaskedPrivateTensor], type(x)
        assert type(y) in [MaskedPrivateTensor], type(y)

        a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped
        b, b0, b1,  beta_on_0,  beta_on_1 = y.unwrapped

        with tf.name_scope('mul'):

            with tf.device(self.crypto_producer.device_name):
                ab = a * b
                ab0, ab1 = _share(ab)

            with tf.device(self.server_0.device_name):
                alpha = alpha_on_0
                beta = beta_on_0
                z0 = ab0 + (a0 * beta) + (alpha * b0) + (alpha * beta)

            with tf.device(self.server_1.device_name):
                alpha = alpha_on_1
                beta = beta_on_1
                z1 = ab1 + (a1 * beta) + (alpha * b1)
        
        z = PrivateTensor(z0, z1)
        z = _truncate(z)
        _nodes[node_key] = z

    return z

def _lift(prot, x):
    # TODO[Morten] support other types of `x`

    if isinstance(x, PublicTensor):
        return x
    
    if isinstance(x, PrivateTensor):
        return x

    if type(x) == int:
        return PublicTensor.from_native(x)

    raise TypeError("Unsupported type {}".format(type(x)))

class Pond(Protocol):

    def __init__(self, server_0, server_1, crypto_producer):
        self.server_0 = server_0
        self.server_1 = server_1
        self.crypto_producer = crypto_producer

    def define_public_placeholder(self, shape, name=None):

        with tf.name_scope('public-placeholder{}'.format('-'+name if name else '')):

            with tf.device(self.server_0.device_name):
                x_on_0 = Placeholder(shape)

            with tf.device(self.server_1.device_name):
                x_on_1 = Placeholder(shape)

        return PublicPlaceholder(self, x_on_0, x_on_1)

    def define_private_placeholder(self, shape, name=None):

        with tf.name_scope('private-placeholder{}'.format('-'+name if name else '')):

            with tf.device(self.server_0.device_name):
                x0 = Placeholder(shape)

            with tf.device(self.server_1.device_name):
                x1 = Placeholder(shape)

        return PrivatePlaceholder(self, x0, x1)

    def define_public_variable(self, initial_value, apply_encoding=True, name=None):

        # TODO[Morten] replace dummy implementation

        assert type(initial_value) in [np.ndarray]
    
        v = initial_value
        v = _encode(v) if apply_encoding else v
        v = Tensor.from_values(v)

        v0, v1 = _share(v)
        assert type(v0) in [Tensor], type(v0)
        assert type(v1) in [Tensor], type(v1)

        with tf.name_scope('var{}'.format('-'+name if name else '')):

            with tf.device(self.server_0.device_name):
                x0 = Variable(v0)

            with tf.device(self.server_1.device_name):
                x1 = Variable(v1)

            x = PublicVariable(self, x0, x1)

        return x

    def define_private_variable(self, initial_value, apply_encoding=True, name=None):
        
        # TODO[Morten] replace dummy implementation

        assert type(initial_value) in [np.ndarray]
    
        v = initial_value
        v = _encode(v) if apply_encoding else v
        v = Tensor.from_values(v)

        v0, v1 = _share(v)
        assert type(v0) in [Tensor], type(v0)
        assert type(v1) in [Tensor], type(v1)

        with tf.name_scope('var{}'.format('-'+name if name else '')):

            with tf.device(self.server_0.device_name):
                x0 = Variable(v0)

            with tf.device(self.server_1.device_name):
                x1 = Variable(v1)

            x = PrivateVariable(self, x0, x1)

        return x

    def assign(self, variable, value):
        assert isinstance(variable, PrivateVariable)
        assert isinstance(value, PrivateTensor)

        var0, var1 = variable.variable0, variable.variable1
        val0, val1 = value.share0, value.share1

        with tf.name_scope('assign'):

            with tf.device(self.server_0.device_name):
                op0 = var0.assign(val0)

            with tf.device(self.server_1.device_name):
                op1 = var1.assign(val1)

        return tf.group(op0, op1)

    def add(x, y):
        assert type(x) in [PrivateTensor], type(x)
        assert type(y) in [PrivateTensor], type(y)
        
        node_key = ('add', x, y)
        z = _nodes.get(node_key, None)

        if z is None:

            x0, x1 = x.unwrapped
            y0, y1 = y.unwrapped
            assert type(x0) in [Int100Tensor], type(x0)
            assert type(x1) in [Int100Tensor], type(x1)
            assert type(y0) in [Int100Tensor], type(y0)
            assert type(y1) in [Int100Tensor], type(y1)
            
            with tf.name_scope('add'):
            
                with tf.device(self.server_0.device_name):
                    z0 = x0 + y0

                with tf.device(self.server_1.device_name):
                    z1 = x1 + y1

            z = PrivateTensor(z0, z1)
            _nodes[node_key] = z

        return z

    def sub(x, y):
        assert type(x) in [PrivateTensor], type(x)
        assert type(y) in [PrivateTensor], type(y)
        
        node_key = ('sub', x, y)
        z = _nodes.get(node_key, None)

        if z is None:

            x0, x1 = x.unwrapped
            y0, y1 = y.unwrapped
            assert type(x0) in [Int100Tensor], type(x0)
            assert type(x1) in [Int100Tensor], type(x1)
            assert type(y0) in [Int100Tensor], type(y0)
            assert type(y1) in [Int100Tensor], type(y1)
            
            with tf.name_scope('sub'):
            
                with tf.device(self.server_0.device_name):
                    z0 = x0 - y0

                with tf.device(self.server_1.device_name):
                    z1 = x1 - y1

            z = PrivateTensor(z0, z1)
            _nodes[node_key] = z

        return z

    def mul(self, x, y):
        # TODO[Morten] apply lifting to x and y
        
        if isinstance(x, PublicTensor) and isinstance(y, PublicTensor):
            return _mul_public_public(self, x, y)

        elif isinstance(x, PublicTensor) and isinstance(y, PrivateTensor):
            return _mul_public_private(self, x, y)
        
        elif isinstance(x, PrivateTensor) and isinstance(y, PublicTensor):
            return _mul_private_public(self, x, y)

        elif isinstance(x, PrivateTensor) and isinstance(y, PrivateTensor):
            return _mul_private_private(self, x, y)

        else:
            raise TypeError("Should not happen")

    def square(self, x):

        node_key = ('square', x)
        y = _nodes.get(node_key, None)

        if y is None:

            if isinstance(x, PrivateTensor):
                x = mask(x)

            assert type(x) in [MaskedPrivateTensor], type(x)
            a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

            with tf.name_scope('square'):

                with tf.device(self.crypto_producer.device_name):
                    aa = a * a
                    aa0, aa1 = share(aa)

                with tf.device(self.server_0.device_name):
                    alpha = alpha_on_0
                    # TODO replace with `scale(, 2)` op
                    y0 = aa0 + (a0 * alpha) + (alpha * a0) + (alpha * alpha)

                with tf.device(self.server_1.device_name):
                    alpha = alpha_on_1
                    # TODO replace with `scale(, 2)` op
                    y1 = aa1 + (a1 * alpha) + (alpha * a1)
            
            y = PrivateTensor(y0, y1)
            y = truncate(y)
            _nodes[node_key] = y

        return y

    def dot(self, x, y):

        node_key = ('dot', x, y)
        z = _nodes.get(node_key, None)

        if z is None:

            if isinstance(x, PrivateTensor):
                x = mask(x)

            if isinstance(y, PrivateTensor):
                y = mask(y)

            assert type(x) in [MaskedPrivateTensor], type(x)
            assert type(y) in [MaskedPrivateTensor], type(y)
            
            a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped
            b, b0, b1,  beta_on_0,  beta_on_1 = y.unwrapped

            with tf.name_scope('dot'):

                with tf.device(self.crypto_producer.device_name):
                    ab = a.dot(b)
                    ab0, ab1 = share(ab)

                with tf.device(self.server_0.device_name):
                    alpha = alpha_on_0
                    beta = beta_on_0
                    z0 = ab0 + a0.dot(beta) + alpha.dot(b0) + beta

                with tf.device(self.server_1.device_name):
                    alpha = alpha_on_1
                    beta = beta_on_1
                    z1 = ab1 + a1.dot(beta) + alpha.dot(b1)

            z = PrivateTensor(z0, z1)
            z = truncate(z)
            _nodes[node_key] = z

        return z

    def sigmoid(self, x):
        assert type(x) in [PrivateTensor], type(x)

        w0 =  0.5
        w1 =  0.2159198015
        w3 = -0.0082176259
        w5 =  0.0001825597
        w7 = -0.0000018848
        w9 =  0.0000000072

        with tf.name_scope('sigmoid'):

            # TODO optimise depth
            x2 = x.square()
            x3 = x2 * x
            x5 = x2 * x3
            x7 = x2 * x5
            x9 = x2 * x7

            y1 = scale(x,  w1)
            y3 = scale(x3, w3)
            y5 = scale(x5, w5)
            y7 = scale(x7, w7)
            y9 = scale(x9, w9)

            # TODO[Morten] verify
            # TODO[Morten] fix last term
            z = y1 + y3 + y5 + y7 + decompose(encode(np.array([w0])))

        return z

class PublicTensor(object):
    
    def __init__(self, prot, value_on_0, value_on_1):
        assert isinstance(value_on_0, Tensor), type(value_on_0)
        assert isinstance(value_on_1, Tensor), type(value_on_1)
        assert value_on_0.shape == value_on_1.shape

        self.prot = prot
        self.value_on_0 = value_on_0
        self.value_on_1 = value_on_1
    
    @property
    def shape(self):
        return self.value_on_0.shape

    def __mul__(self, other):
        self.protocol.mul(self, other)

class PrivateTensor(object):
    
    def __init__(self, prot, share0, share1):
        assert isinstance(share0, Tensor), type(share0)
        assert isinstance(share1, Tensor), type(share1)
        assert share0.shape == share1.shape

        self.prot = prot
        self.share0 = share0
        self.share1 = share1
    
    @property
    def shape(self):
        return self.share0.shape

    @property
    def unwrapped(self):
        return (self.share0, self.share1)
    
    def __add__(self, other):
        return self.prot.add(self, other)

    def __sub__(self, other):
        return self.prot.sub(self, other)

    def __mul__(self, other):
        return self.prot.mul(self, other)

    def add(self, other):
        return self.prot.add(self, other)
    
    def sub(self, other):
        return self.prot.sub(self, other)

    def mul(self, other):
        return self.prot.mul(self, other)

    def dot(self, other):
        return self.prot.dot(self, other)

    def transpose(self):
        return self.prot.transpose(self)

    def truncate(self):
        return self.prot.truncate(self)

class MaskedPrivateTensor(object):

    def __init__(self, x, a, a0, a1, alpha_on_0, alpha_on_1):
        assert isinstance(x, PrivateTensor)
        self.x  = x
        self.a  = a
        self.a0 = a0
        self.a1 = a1
        self.alpha_on_0 = alpha_on_0
        self.alpha_on_1 = alpha_on_1

    @property
    def shape(self):
        return self.a.shape

    @property
    def unmasked(self):
        return self.x

    @property
    def unwrapped(self):
        return (self.a, self.a0, self.a1, self.alpha_on_0, self.alpha_on_1)

class PublicPlaceholder(PublicTensor):

    def __init__(self, placeholder_on_0, placeholder_on_1):
        assert type(placeholder_on_0) in [Placeholder], type(placeholder_on_0)
        assert type(placeholder_on_1) in [Placeholder], type(placeholder_on_1)
        assert placeholder_on_0.shape == placeholder_on_1.shape
        
        super(PublicPlaceholder, self).__init__(placeholder_on_0, placeholder_on_1)
        self.placeholder_on_0 = placeholder_on_0
        self.placeholder_on_1 = placeholder_on_1

class PrivatePlaceholder(PrivateTensor):

    def __init__(self, prot, placeholder0, placeholder1):
        assert type(placeholder0) in [Placeholder], type(placeholder0)
        assert type(placeholder1) in [Placeholder], type(placeholder1)
        assert placeholder0.shape == placeholder1.shape
        
        super(PrivatePlaceholder, self).__init__(prot, placeholder0, placeholder1)
        self.placeholder0 = placeholder0
        self.placeholder1 = placeholder1

class PublicVariable(PublicTensor):
    pass

class PrivateVariable(PrivateTensor):

    def __init__(self, variable0, variable1):
        assert type(variable0) in [Int100Variable], type(variable0)
        assert type(variable1) in [Int100Variable], type(variable1)
        assert variable0.shape == variable1.shape
        super(PrivateVariable, self).__init__(variable0.backing, variable1.backing)
        self.variable0 = variable0
        self.variable1 = variable1
        self.initializer = tf.group([ var.initializer ] for var in [variable0, variable1])

    @property
    def initializer(self):
        return self.initializer