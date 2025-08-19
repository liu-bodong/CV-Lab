# Utilities

# Acknowledgements:
# MetricDict and Metric classes are adapted from 
# https://github.com/CuriousAI/mean-teacher

import sys

__all__ = ['MetricDict', 'Metric']

class MetricDict:
    def __init__(self):
        self.dict = {}

    def __getitem__(self, key):
        return self.dict[key]

    def update(self, key, value, n=1):
        if not key in self.dict:
            self.dict[key] = Metric()
        self.dict[key].update(value, n)

    def reset(self):
        for value in self.dict.values():
            value.reset()

    def values(self, postfix=''):
        return {key + postfix: value.val for key, value in self.dict.items()}

    def averages(self, postfix='/avg'):
        return {key + postfix: value.avg for key, value in self.dict.items()}

    def sums(self, postfix='/sum'):
        return {key + postfix: value.sum for key, value in self.dict.items()}

    def counts(self, postfix='/count'):
        return {key + postfix: value.count for key, value in self.dict.items()}

class Metric:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)