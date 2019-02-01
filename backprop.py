import numpy as np
import math
from demo_linear_predict import *

ALPHA = 0.001

def valid(val):
    return not np.isinf(val) and not np.isnan(val)

class F:
    def __init__(self, *args):
        pass
    def evaluate(self):
        pass
    def backprop(self, g):
        pass

class VAR(F):
    def __init__(self, name = "", ival = 0, const = True):
        self.value = ival
        self.const = const
        self.name = name
    def set(self, newval):
        self.value = newval
    def evaluate(self):
        return self.value
    def backprop(self, g):
        if not self.const:
            new_val = self.value - ALPHA * g
            assert(valid(new_val))
            self.value = new_val

class W(VAR):
    def __init__(self, name = "", ival = 0):
        self.value = ival
        self.const = False
        self.name = name

class CONST(VAR):
    def __init__(self, name = "", ival = 0):
        self.value = ival
        self.const = True
        self.name = name

class MULT(F):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.const = False
        self.name = "({}*{})".format(a.name, b.name)
    def evaluate(self):
        return self.a.evaluate() * self.b.evaluate()
    def backprop(self, g):
        if not self.a.const:
            self.a.backprop(g * self.b.evaluate())
        if not self.b.const:
            self.b.backprop(g * self.a.evaluate())

class ADD(F):
    def __init__(self, *args):
        self.vars = args
        self.const = False
        self.name = "(" + "+".join([v.name for v in args]) + ")"
    def evaluate(self):
        return sum([v.evaluate() for v in self.vars])
    def backprop(self, g):
        for v in self.vars:
            if not v.const:
                v.backprop(g)

class SUB(F):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.const = False
        self.name = "({}-{})".format(a.name, b.name)
    def evaluate(self):
        return self.a - self.b
    def backprop(self, g):
        if not self.a.const:
            self.a.backprop(g)
        if not self.b.const:
            self.b.backprop(-g)

class MAX(F):
    def __init__(self, *args):
        self.vars = args
        self.const = False
    def evaluate(self):
        return max([v.evaluate() for v in self.vars])
    def backprop(self, g):
        i = np.argmax([v.evaluate() for v in self.vars])
        v = self.vars[i]
        if not v.const:
            v.backprop(g)

class SIGMOID(F):
    def __init__(self, v):
        self.v = v
        self.const = False
    def evaluate(self):
        return 1/(1+math.exp(-self.v.evaluate()))
    def backprop(self, g):
        i = np.argmax([v.evaluate() for v in self.vars])
        v = self.vars[i]
        if not v.const:
            v.backprop(g)

class POW(F):
    def __init__(self, v, p):
        self.v = v
        self.p = p
        self.const = False
        self.name = "({}**{})".format(v.name, p)
    def evaluate(self):
        return math.pow(self.v.evaluate(), self.p)
    def backprop(self, g):
        if not self.v.const:
            self.v.backprop(g * self.p * math.pow(self.v, self.p - 1))

class DOT(F):
    def __init__(self, vdict = {}, wdict = {}):
        self.v = vdict
        self.w = wdict
        self.const = False
        self.name = "(DOT(...))"
    def set(self, vdict={}, wdict={}):
        self.v = vdict
        self.w = wdict
    def set_var(self, key, value):
        self.v[key] = value
    def set_weight(self, key, value):
        self.w[key] = value
    def evaluate(self):
        r = 0
        for (k, v) in self.v.items():
            if k in self.w:
                r = r + v * self.w[k]
        return r
    def backprop(self, g):
        res = {}
        for (k, v) in self.v.items():
            if k not in self.w:
                new_val = - ALPHA * g * v
            else:
                new_val = self.w[k] - ALPHA * g * v
            assert (valid(new_val))
            res[k] = new_val
        self.w = res


def get_target(a, b):
    return a.evaluate() * 10 - b.evaluate() * 5 + np.random.normal(loc = 0, scale = 0.001)

def get_target_scalar(a, b):
    return a * 10 - b * 5 + np.random.normal(loc = 0, scale = 0.001)

def simple_test1():
    a = VAR()
    b = VAR()
    w1 = W()
    w2 = W()

    l1 = ADD(MULT(w1, a), MULT(b, w2))

    for i in range(1000):
        a.set(np.random.uniform(-3, 3))
        b.set(np.random.uniform(-3, 3))

        target = get_target(a, b)
        prediction = l1.evaluate()

        error = -(target - prediction)
        l1.backprop(error)

    print("w1 = {}, w2 = {} ".format(w1.evaluate(), w2.evaluate()))


def simple_test2():
    data = generate_training_data(1000, True)

    x1 = VAR("x1")
    x2 = VAR("x2")
    x1_square = POW(x1, 2)
    x2_square = POW(x2, 2)
    bias = CONST("b", 1)

    phi = [x1, x2, x1_square, x2_square, bias]
    weights = [W("w" + str(i + 1)) for i in range(5)]

    ops = [MULT(xi, wi) for (xi, wi) in zip(phi, weights)]
    graph = ADD(*ops)

    for i in range(10):
        for p in data:
            x = p[0]
            x1.set(x[0])
            x2.set(x[1])

            target = p[1]
            prediction = graph.evaluate()

            error = -(target - prediction)
            graph.backprop(error)

    wvals = [w.evaluate() for w in weights]
    generate_prediction(wvals)

def simple_test3():
    graph = DOT()
    for i in range(100000):
        graph.set_var("x1", np.random.uniform(-3, 3))
        graph.set_var("x2", np.random.uniform(-3, 3))
        target = get_target_scalar(graph.v["x1"], graph.v["x2"])
        prediction = graph.evaluate()
        error = -(target - prediction)
        graph.backprop(error)

    print("w1 = {}, w2 = {} ".format(graph.w["x1"], graph.w["x2"]))


if __name__ == "__main__":
    # simple_test1()
    # simple_test2()
    simple_test3()