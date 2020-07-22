import os
import glob
import importlib
from .utils import get_model, avail_models, register_model

_here = os.path.dirname(os.path.abspath(__file__))
_pyfiles = glob.glob(os.path.join(_here, "*.py"))
for f in _pyfiles:
    mod_name = os.path.basename(f).split(".")[0]
    if mod_name == "__init__":
        continue
    importlib.import_module("." + mod_name, __name__)

__all__ = ["get_model", "avail_models", "register_model"]

"""
The codes of some basic models under this directory are copy and modified from https://github.com/kuangliu/pytorch-cifar/commit/340751189c307d91e243df26d6d5779b7a29f781
Here is the license of that code repo:
------------
MIT License

Copyright (c) 2017 liukuang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
