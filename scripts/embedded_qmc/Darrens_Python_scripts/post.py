#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:54:14 2021

@author: darrenlean
"""

import matplotlib.pyplot as plt
import numpy
import os
from cisd_coeff import Hamiltonian, RestoredCisdCoeffs, compare_two_cisd
import pickle as pkl

params = {
   'axes.labelsize': 10,
   'font.size': 12,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'figure.figsize': [10, 5.25]
   } 
plt.rcParams.update(params)

class BenchmarkSingleRun:
    def __init__(self, folder_name):
        self._f = folder_name
        self.ham = Hamiltonian()
        
        # Loading Hamiltonian from the pickle
        self.ham.from_pickle(self._f + '/hamiltonian.pkl')
        self.cisd_coeffs = RestoredCisdCoeffs(self.ham)
        self.exact = RestoredCisdCoeffs(self.ham)
        
    def get_exact_solution(self):
        solution  = self.exact.get_fci()
        self.fci_energy = solution[0]
        self.ref_energy = self.exact.ref_energy()
        self.correlation_energy = self.ref_energy - self.fci_energy
        self.exact.from_fcivec(solution[1])
        self.exact.normalise()
    
    def get_h5_name(self):
        for file in os.listdir(self._f):
            if file.endswith('.h5'):
                return file
        return None
    
    def process_data(self):
        h5_name = self.get_h5_name()
        # Getting CISD coefficients from M7 output
        self.cisd_coeffs.from_m7(self._f + '/' + h5_name)
        self.cisd_coeffs.normalise()
        self.h5_energy = self.cisd_coeffs.energy()
        self.h5_cosine_similarity = compare_two_cisd(self.cisd_coeffs, self.exact)[1]
        
    def print_results(self):
        print("FCI energy          :", self.fci_energy)
        print("h5 energy           :", self.h5_energy)
        print("h5 cosine similarity:", self.h5_cosine_similarity)

class PostProcessing:
    
    def __init__(self, folder_name, header="nw="):
        self._f = folder_name
        self._files = []
        self.header = header
        self.get_files()
        self.nw = []
        self.ham = Hamiltonian()

        # Loading Hamiltonian from the pickle
        self.ham.from_pickle(self._f + '/hamiltonian.pkl')
        self.cisd_coeffs = RestoredCisdCoeffs(self.ham)
        self.exact = RestoredCisdCoeffs(self.ham)
        
    def get_exact_solution(self):
        solution  = self.exact.get_fci()
        self.fci_energy = solution[0]
        self.ref_energy = self.exact.ref_energy()
        self.correlation_energy = self.ref_energy - self.fci_energy
        self.exact.from_fcivec(solution[1])
        self.exact.normalise()
    
    def get_files(self):
        self._files = []
        temporary = []
        for file in os.listdir(self._f+'/'):
            if file.startswith(self.header):
                temporary.append(int(file[len(self.header):]))
        temporary.sort()
        for num in temporary:
            self._files.append(self.header+str(num))
        return self._files
    
    def get_nw(self, file): 
        f = numpy.loadtxt(self._f + '/' + file + '/M7.stats')
        array = f[:,3]
        left = self.left_bracket(array, 100, 0.05)
        return numpy.average(array[left:])
    
    def get_names(self, file, ending='.h5'):
        names = []
        for file in os.listdir(self._f + '/' + file):
            if file.endswith(ending):
                names.append(file)
        names.sort()
        return names
    
    def left_bracket(self, array, window = 10, tolerance = 0.0005):
        '''
        A windowing function that takes in an array to be windowed, the size of 
        the window and a tolerance on the percentage error
        This function windows 'from the left to the right'
        Returns the 'left hand side' index of the window
        '''
        array_avg = numpy.average(array)
        array_std = numpy.std(array)

        i = 0
        while abs(array_std/array_avg) > tolerance:
            i += 10
            array_avg = numpy.average(array[i:i+window])
            array_std = numpy.std(array[i:i+window])
        return i    

    def process_h5(self, file, store_coeff = False):
        names = self.get_names(file, ending='.h5')
        e_temporary = []
        c_temporary = []
        self.c0 = None
        self.c1a = None
        self.c1b = None
        self.c2aa = None
        self.c2bb = None
        self.c2ab = None
        for name in names:
            # Getting CISD coefficients from M7 output
            self.cisd_coeffs.from_m7(self._f + '/' + file + '/' + name)
            assert self.cisd_coeffs.c0 != 0, file + "/" + name + " has zero reference weight (c0 = 0)"
            self.cisd_coeffs.normalise()
            e_temporary.append(self.cisd_coeffs.energy())
            c_temporary.append(compare_two_cisd(self.cisd_coeffs, self.exact)[1])
            if store_coeff:
                if self.c0 == None:
                    self.c0 = self.cisd_coeffs.c0
                    self.c1a = self.cisd_coeffs.c1a
                    self.c1b = self.cisd_coeffs.c1b
                    self.c2aa = self.cisd_coeffs.c2aa
                    self.c2bb = self.cisd_coeffs.c2bb
                    self.c2ab = self.cisd_coeffs.c2ab
                else:
                    self.c0 += self.cisd_coeffs.c0
                    self.c1a += self.cisd_coeffs.c1a
                    self.c1b += self.cisd_coeffs.c1b
                    self.c2aa += self.cisd_coeffs.c2aa
                    self.c2bb += self.cisd_coeffs.c2bb
                    self.c2ab += self.cisd_coeffs.c2ab
        if self.c0 != None:
            self.c0 /= len(names)
            self.c1a /= len(names)
            self.c1b /= len(names)
            self.c2aa /= len(names)
            self.c2bb /= len(names)
            self.c2ab /= len(names)
        self.energy_frac_sys_err.append(abs(numpy.average(e_temporary)-self.fci_energy)/self.correlation_energy)
        self.energy_frac_ran_err.append(numpy.sqrt(numpy.var(e_temporary)/len(e_temporary))/self.correlation_energy)
        self.coeff_sys_err.append(numpy.average(c_temporary))
        self.coeff_ran_err.append(numpy.sqrt(numpy.var(c_temporary)/len(c_temporary)))
        return numpy.average(e_temporary)
        
    def process_data(self):
        self.nw = []
        self.energy_frac_ran_err = []
        self.energy_frac_sys_err = []
        self.coeff_ran_err = []
        self.coeff_sys_err = []
        for file in self._files:
            self.nw.append(self.get_nw(file))
            if file == self._files[-1]:
                self.process_h5(file, True)
            else:
                self.process_h5(file, False)
        
    def show_graphs(self):
        plt.subplot(211)
        plt.title("Systematic improvement of FCIQMC-estimated reference excitation amplitudes " + self._f)
        plt.axhline(0, color='black', ls='--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of walkers')
        plt.xlabel('Size of semi-stochastic space')
        plt.ylabel(r'Energy error as a fraction of $E_{corr}$')
        plt.errorbar(self.nw, self.energy_frac_sys_err, \
                     yerr=self.energy_frac_ran_err, capsize=5)
        
        plt.subplot(212)
        plt.axhline(0, color='black', ls='--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of walkers')
        plt.xlabel('Size of semi-stochastic space')
        plt.ylabel(r"$C_1$, $C_2$ cosine similarity")
        plt.errorbar(self.nw, self.coeff_sys_err, \
                     yerr=self.coeff_ran_err, capsize=5)
    
    def coeff_to_pickle(self, fname = None):
        if fname == None:
            fname = self._f + "/avg_coeff.pkl"
        with open(fname, 'wb') as f: pkl.dump([self.c0, self.c1a, self.c1b, self.c2aa, self.c2bb, self.c2ab], f)
        
class PostProcessing_SS:
    
    def __init__(self, folder_name):
        self._f = folder_name
        self.ham = Hamiltonian()

        # Loading Hamiltonian from the pickle
        self.ham.from_pickle(self._f + '/hamiltonian.pkl')
        self.cisd_coeffs = RestoredCisdCoeffs(self.ham)
        self.exact = RestoredCisdCoeffs(self.ham)
        
    def get_exact_solution(self):
        solution  = self.exact.get_fci()
        self.fci_energy = solution[0]
        self.ref_energy = self.exact.ref_energy()
        self.correlation_energy = self.ref_energy - self.fci_energy
        self.exact.from_fcivec(solution[1])
        self.exact.normalise()
    
    def get_files(self, path, header):
        result = []
        temporary = []
        for file in os.listdir(self._f+'/'+path):
            if file.startswith(header):
                temporary.append(int(file[len(header):]))
        temporary.sort()
        for num in temporary:
            result.append(header+str(num))
        return result
    
    def get_nw(self, file): 
        f = numpy.loadtxt(self._f + '/' + file + '/M7.stats')
        array = f[:,3]
        left = self.left_bracket(array, 100, 0.05)
        return numpy.average(array[left:])
    
    def get_var_shift_idx(self, file):
        f = numpy.loadtxt(self._f + '/' + file + '/M7.stats')
        array = f[:,2]
        for i in range(len(array)-1):
            if array[i+1] != array[i]:
                return i+1
    
    def get_delay(self, file):
        property_header = "av_ests.delay: "
        delay = 0
        with open(self._f + '/' + file + '/M7.log') as f:
            lines = f.readlines()
        truncate = 0
        while lines[0][truncate:truncate+7]!= "[info] ":
            truncate += 1
        truncate += 7
        for line in lines:
            if line[truncate:truncate+len(property_header)] == property_header:
                delay = int(line[truncate+len(property_header):])
        return delay
    
    def process_stats(self, file):
        f = numpy.loadtxt(self._f + '/' + file + '/M7.stats')
        N = self.get_var_shift_idx(file)
        delay = self.get_delay(file)
        numerator_array = f[:,7][N+delay:]
        denominator_array = f[:,8][N+delay:]
        numerator_mean = numpy.average(numerator_array)
        numerator_std = numpy.sqrt(numpy.var(numerator_array)/len(numerator_array))
        denominator_mean = numpy.average(denominator_array)
        denominator_std = numpy.sqrt(numpy.var(denominator_array)/len(denominator_array))
        mean_energy = numerator_mean/denominator_mean
        std_energy = numpy.sqrt((numerator_std/denominator_mean)**2 + \
                    (numerator_mean*denominator_std/denominator_mean**2)**2)
        return [mean_energy, std_energy, abs(mean_energy-self.fci_energy)]
    
    def get_names(self, file, ending='.h5'):
        names = []
        for file in os.listdir(self._f + '/' + file):
            if file.endswith(ending):
                names.append(file)
        names.sort()
        return names
    
    def left_bracket(self, array, window = 10, tolerance = 0.0005):
        '''
        A windowing function that takes in an array to be windowed, the size of 
        the window and a tolerance on the percentage error
        This function windows 'from the left to the right'
        Returns the 'left hand side' index of the window
        '''
        array_avg = numpy.average(array)
        array_std = numpy.std(array)

        i = 0
        while abs(array_std/array_avg) > tolerance:
            i += 10
            array_avg = numpy.average(array[i:i+window])
            array_std = numpy.std(array[i:i+window])
        return i    

    def process_h5(self, file):
        h5_results = []
        names = self.get_names(file, ending='.h5')
        n = len(names)
        e_temporary = []
        c_temporary = []
        for name in names:
            # Getting CISD coefficients from M7 output
            self.cisd_coeffs.from_m7(self._f + '/' + file + '/' + name)
            self.cisd_coeffs.normalise()
            e_temporary.append(self.cisd_coeffs.energy())
            c_temporary.append(compare_two_cisd(self.cisd_coeffs, self.exact)[1])
        h5_results.append(numpy.average(e_temporary))
        h5_results.append(numpy.sqrt(numpy.var(e_temporary)/n))
        h5_results.append(abs(e_temporary[0]-self.fci_energy))
        h5_results.append(abs(numpy.average(e_temporary)-self.fci_energy)/self.correlation_energy)
        h5_results.append(numpy.sqrt(numpy.var(e_temporary)/n)/self.correlation_energy)
        h5_results.append(numpy.average(c_temporary))
        h5_results.append(numpy.sqrt(numpy.var(c_temporary)/n))
        return h5_results
        
    def process_data(self):
        header_lv1 = "nw="
        header_lv2 = "size="
        files_lv1 = self.get_files("", header_lv1)
        self.nws = []
        self.sizes = []
        self.e_frac_ran_err = []
        self.e_frac_sys_err = []
        self.c_ran_err = []
        self.c_sys_err = []
        self.stats_e = []
        self.stats_e_ran_err = []
        self.stats_e_sys_err = []
        self.h5_e = []
        self.h5_e_ran_err = []
        self.h5_e_sys_err = []
        for file_lv1 in files_lv1:
            self.nws.append(int(file_lv1[len(header_lv1):]))
            sizes_t = []
            e_frac_ran_err_t = []
            e_frac_sys_err_t = []
            c_ran_err_t = []
            c_sys_err_t = []
            stats_e_t = []
            stats_e_ran_err_t = []
            stats_e_sys_err_t = []
            h5_e_t = []
            h5_e_ran_err_t = []
            h5_e_sys_err_t = []
            files_lv2 = self.get_files(file_lv1, header_lv2)
            for file_lv2 in files_lv2:
                path = file_lv1 + '/' + file_lv2
                h5_results = self.process_h5(path)
                stats_results = self.process_stats(path)
                sizes_t.append(int(file_lv2[len(header_lv2):]))
                e_frac_ran_err_t.append(h5_results[4])
                e_frac_sys_err_t.append(h5_results[3])
                c_ran_err_t.append(h5_results[6])
                c_sys_err_t.append(h5_results[5])
                stats_e_t.append(stats_results[0])
                stats_e_ran_err_t.append(stats_results[1])
                stats_e_sys_err_t.append(stats_results[2])
                h5_e_t.append(h5_results[0])
                h5_e_ran_err_t.append(h5_results[1])
                h5_e_sys_err_t.append(h5_results[2])
            self.sizes.append(sizes_t)
            self.e_frac_ran_err.append(e_frac_ran_err_t)
            self.e_frac_sys_err.append(e_frac_sys_err_t)
            self.c_ran_err.append(c_ran_err_t)
            self.c_sys_err.append(c_sys_err_t)
            self.stats_e.append(stats_e_t)
            self.stats_e_ran_err.append(stats_e_ran_err_t)
            self.stats_e_sys_err.append(stats_e_sys_err_t)
            self.h5_e.append(h5_e_t)
            self.h5_e_ran_err.append(h5_e_ran_err_t)
            self.h5_e_sys_err.append(h5_e_sys_err_t)
        
    def show_graphs(self):
        plt.figure(1)
        plt.subplot(211)
        plt.title("Systematic improvement of FCIQMC-estimated reference excitation amplitudes " + self._f)
        plt.axhline(0, color='black', ls='--')
        plt.xscale('log')
        plt.xlabel('Size of semi-stochastic space')
        plt.ylabel(r'Energy error as a fraction of $E_{corr}$')
        for i in range(len(self.nws)):
            plt.errorbar(self.sizes[i], self.e_frac_sys_err[i], \
                         yerr=self.e_frac_ran_err[i], capsize=5, \
                         label="nw="+str(self.nws[i]))
        plt.legend()
        
        plt.subplot(212)
        plt.axhline(0, color='black', ls='--')
        plt.xscale('log')
        plt.xlabel('Size of semi-stochastic space')
        plt.ylabel(r"$C_1$, $C_2$ cosine similarity")
        for i in range(len(self.nws)):
            plt.errorbar(self.sizes[i], self.c_sys_err[i], \
                         yerr=self.c_ran_err[i], capsize=5, \
                         label="nw="+str(self.nws[i]))
        plt.legend()
        
        plt.figure(2)
        plt.xscale('log')
        plt.xlabel('Size of semi-stochastic space')
        plt.ylabel('Energy')
        plt.axhline(self.fci_energy, color='black', ls='--', label='FCI')
        for i in range(len(self.nws)):
            plt.errorbar(self.sizes[i], self.h5_e[i], yerr=self.h5_e_ran_err[i], \
                         label="h5 nw="+str(self.nws[i]), capsize=5)
        plt.legend()
        
        
        plt.figure(3)
        plt.xscale('log')
        plt.xlabel('Size of semi-stochastic space')
        plt.ylabel('Energy random error')
        plt.axhline(0, color='black', ls='--')
        for i in range(len(self.nws)):
            plt.plot(self.sizes[i], self.stats_e_ran_err[i], label='stats nw='+str(self.nws[i]))
            plt.plot(self.sizes[i], self.h5_e_ran_err[i], label='h5 nw='+str(self.nws[i]))
        plt.legend()
        
        plt.figure(4)
        plt.xscale('log')
        plt.xlabel('Size of semi-stochastic space')
        plt.ylabel('Energy systematic error')
        plt.axhline(0, color='black', ls='--')
        for i in range(len(self.nws)):
            plt.plot(self.sizes[i], self.stats_e_sys_err[i], label='stats nw='+str(self.nws[i]))
            plt.plot(self.sizes[i], self.h5_e_sys_err[i], label='h5 nw='+str(self.nws[i]))
        plt.legend() 
        