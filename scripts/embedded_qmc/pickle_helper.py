#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:48:06 2021

@author: darrenlean
"""

import pickle as pkl

def coeff_from_pickle(fname):
    with open(fname, 'rb') as f:
        c0, c1a, c1b, c2aa, c2bb, c2ab = pkl.load(f)
    return c0, c1a, c1b, c2aa, c2bb, c2ab