#!usr/bin/python
import math
import time
types = {'Step': 0, 'Linear': 1, 'Sigmoid': 2}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Synapse:
    weight = 1.0
    decay = 1
    learn = False
    active = False
    value = 0.0

    def __init__(self, neu):
        self.neu = neu
        self.last = 0.0

    def count(self):
        val_w = self.neu.value * self.weight
        if self.decay == 0:
            self.value = val_w
        else:
            if self.active:
                self.last = self.last * self.decay
                if math.fabs(self.last) <= 0.01:
                    self.active = False
                    self.last = 0
                if math.fabs(self.last) > math.fabs(val_w):
                    self.last = val_w
                self.value = self.last
            else:
                if val_w != 0:
                    self.active = True
                    self.last = val_w
                    self.value = self.last
                else:
                    self.value = 0
        return self.value


class Neuron:
    threshold = 0.5
    synapses = set()
    learn = False
    value = 0
    new_value = 0

    def __init__(self, the_type):
        self.__type = types[the_type]

    def count(self):
        _sum = 0
        for s in self.synapses:
            _sum += s.count()
        if self.__type == types['Step']:
            if _sum >= self.threshold:
                self.new_value = 1
            else:
                self.new_value = 0
        elif self.__type == types['Linear']:
            self.new_value = _sum
        else:
            self.new_value = sigmoid(_sum)

    def ok(self):
        self.value = self.new_value


a = Neuron('Step')
b = Neuron('Step')
a.threshold = 0
b.threshold = 0
a2b = Synapse(a)
b2a = Synapse(b)
a2b.weight = -1
b2a.weight = -1
a2b.decay = 0.5
b2a.decay = 0.5
a.synapses.add(b2a)
b.synapses.add(a2b)
for i in range(10):
    a.count()
    b.count()
    a.ok()
    b.ok()
    print("a: {}  b: {}  a2b: {}  b2a: {}".format(a.value, b.value, a2b.value, b2a.value))
    # time.sleep(0.5)
