#!usr/bin/python
import math
types = {'Step': 0, 'Linear': 1, 'Sigmoid': 2}
t = 0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Synapse:
    def __init__(self, neu):
        self.neu = neu
        self.weight = 1
        self.decay = 1
        self.learn = False
        self.active = False
        self.decayed = False
        self.last = 0
        self.__time = 0

    def value(self):
        update = self.__time != t
        self.__time = t
        val_w = self.neu.value * self.weight
        if self.decay == 0:
            return val_w
        else:
            if self.active:
                if update:
                    self.last = self.last * self.decay
                if math.fabs(self.last) > math.fabs(val_w):
                    self.last = val_w
                if math.fabs(self.last) <= 0.01:
                    self.active = False
                    self.decayed = True
                    self.last = 0
                return self.last
            else:
                if val_w != 0:
                    if not self.decayed:
                        self.active = True
                        self.last = val_w
                        return self.last
                    else:
                        return self.last
                else:
                    self.decayed = False
                    self.last = val_w
                    return self.last


class InputNode:
    def __init__(self):
        self.value = 0


class Neuron:
    def __init__(self, the_type):
        self.__type = types[the_type]
        self.threshold = 0.5
        self.synapses = set()
        self.learn = False
        self.value = 0

    def count(self):
        s_sum = 0
        for s in self.synapses:
            s_sum += s.value()
        if self.__type == types['Step']:
            if s_sum >= self.threshold:
                self.value = 1
            else:
                self.value = 0
        elif self.__type == types['Linear']:
            self.value = s_sum
        else:
            self.value = sigmoid(s_sum)


# threshold and weight should be in [-1, 1]
# decay should be in [0, 1]
# the larger decay is, the faster the decay are.

def test1():
    global t
    a = Neuron('Step')
    b = Neuron('Step')
    a.threshold = 0
    b.threshold = 0
    a2b = Synapse(a)
    b2a = Synapse(b)
    a2b.weight = -1
    b2a.weight = -1
    a2b.decay = 0.2
    b2a.decay = 0.2
    a.synapses.add(b2a)
    b.synapses.add(a2b)
    for i in range(20):
        t += 1
        a.count()
        b.count()
        a.count()
        print("a: {}  b: {}".format(a.value, b.value))


def test2():
    global t
    i = InputNode()
    a = Neuron('Step')
    i2a = Synapse(i)
    a.synapses.add(i2a)
    a2a = Synapse(a)
    a2a.decay = 0.9
    a.synapses.add(a2a)
    for _ in range(20):
        t += 1
        if _ == 5:
            i.value = 1
        if _ == 7:
            i.value = 0
        a.count()
        print("i: {}  a: {}".format(i.value, a.value))


def test3():
    global t
    a = Neuron('Step')
    b = Neuron('Step')
    i = InputNode()
    i2a = Synapse(i)
    a2a = Synapse(a)
    a2b = Synapse(a)
    a2bn = Synapse(a)
    a2b.weight = 0.7
    a2bn.weight = -1
    a2a.decay = 0.9
    a2bn.decay = 0.7
    a.synapses.add(i2a)
    a.synapses.add(a2a)
    b.synapses.add(a2b)
    b.synapses.add(a2bn)
    for _ in range(20):
        t += 1
        if _ == 2:
            i.value = 1
        if _ == 4:
            i.value = 0
        a.count()
        b.count()
        print("i: {}  out: {}".format(i.value, b.value))


test3()
