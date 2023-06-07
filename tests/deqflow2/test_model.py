import pytest
import torch
import deqflow2

def test_train(mod_deqflow):
    assert isinstance(mod_deqflow, deqflow2.DEQFlow)

def test_eval(mod_deqflow):
    assert isinstance(mod_deqflow, deqflow2.DEQFlow)
