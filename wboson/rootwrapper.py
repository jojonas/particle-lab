import contextlib

import ROOT

@contextlib.contextmanager
def root_open(filename, *args, **kwargs):
    file = ROOT.TFile(filename, *args, **kwargs)
    yield file
    file.Close()
