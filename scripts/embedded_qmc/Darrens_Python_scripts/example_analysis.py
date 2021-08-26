#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:07:58 2021

@author: darrenlean
"""

from post import BenchmarkSingleRun, PostProcessing, PostProcessing_SS


'''
2D Hubbard model example
'''

# Working directory
directory = "embedded_qmc/1D Hubbard U=2/frag0"

test = BenchmarkSingleRun(directory)
#test = PostProcessing(directory)
test.get_exact_solution()
test.process_data()
#test.show_graphs()
#test.coeff_to_pickle()
test.print_results()
