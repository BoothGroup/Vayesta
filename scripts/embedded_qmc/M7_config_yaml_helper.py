#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:06:39 2021

@author: darrenlean
"""

from subprocess import Popen, PIPE

def write_dictionary(file, dictionary, n_of_indentation=0):
    for k in dictionary.keys():
        if type(dictionary[k]) != dict:
            file.write(n_of_indentation*"  "+"%s: %s\n" % (k, dictionary[k]))
        else:
            file.write(n_of_indentation*"  "+"%s:\n" % k)
            n_of_indentation += 1
            write_dictionary(file, dictionary[k], n_of_indentation)
            n_of_indentation -= 1
    
def dict_to_yaml(file, dictionary):
    with open(file, 'w') as f:
        f.write("---\n")
        write_dictionary(f, dictionary)

class M7_config_to_dict:
    
    def __init__(self, path_to_M7):
        self.path_to_M7 = path_to_M7
        process = Popen('./release | less -RC', stdout=PIPE, stderr=PIPE, \
                        shell=True, cwd=self.path_to_M7+'/build/src')
        self.stdout, self.stderr = process.communicate()
        self.stdout = self.stdout.decode()
        self.M7_config_dict = {}
        self.make_M7_config_dict()
        
    def find_all_indices_of_substring(self, string, substring):
        idx = []
        start = 0
        while string.find(substring, start) != -1:
            idx.append(string.find(substring, start))
            start = string.find(substring, start) + 1
        return idx
    
    def get_level(self, string, index):
        count_space = 0
        index -= 1
        while string[index] == " ":
            count_space += 1
            index -=1
        return count_space//2
    
    def find_name(self, string):
        return string[string.find('\x1b[1m') + len('\x1b[1m'):string.find('\x1b[0m')]
    
    def find_default_value(self, string):
        start = string.find('value:') + len('value:')
        end = string.find('Description:')
        value = string[start:end]
        value = value.replace("\n", "")
        value = value.replace(" ", "")
        if value != '""':
            value = value.replace('"', '')
        if value.isnumeric():
            return int(value)
        elif value.count('.') == 1 and value.replace('.', '').isnumeric():
            return float(value)
        if value == 'inf':
            return 18446744073709551615
        return value
    
    def update_current_dict(self):
        self.current_dict = self.M7_config_dict
        for k in self.current_directory:
            self.current_dict = self.current_dict[k]
    
    def make_nested_dict(self, header, child_dict=None, parameter=None, value=None):
        if child_dict == None:
            assert parameter != None
            assert value != None
            return {header: {parameter: value}}
        else:
            assert parameter == None
            assert value == None
            return {header: child_dict}
    
    
    def make_M7_config_dict(self):
        section_idx = self.find_all_indices_of_substring(self.stdout, 'Section')
        parameter_idx = self.find_all_indices_of_substring(self.stdout, 'Parameter')   
        
        self.M7_config_dict = {}
        self.current_dict = self.M7_config_dict
        new_section = None
        self.current_directory = []
        for i in range(len(self.stdout)):
            if i in section_idx:
                section_name = self.find_name(self.stdout[i:])
                n = self.get_level(self.stdout, i)
                assert section_name.count('.') == n, "Section level and indentation mismatch"
                while section_name.count('.') != 0:
                    section_name = section_name[section_name.find('.')+1:]
                if n < len(self.current_directory):
                    while n != len(self.current_directory):
                        self.current_directory.pop(-1)
                    self.update_current_dict()
                if new_section == None:
                    new_section = section_name
                elif type(new_section) == list:
                    new_section.append(section_name)
                else:
                    new_section = [new_section, section_name]
            if i in parameter_idx:
                parameter_name = self.find_name(self.stdout[i:])
                default_value = self.find_default_value(self.stdout[i:])
                n = self.get_level(self.stdout, i)
                if new_section:
                    if type(new_section) == list:
                        for j in reversed(range(len(new_section))):
                            if j == len(new_section)-1:
                                temp_dict = self.make_nested_dict(new_section[j], child_dict=None, parameter=parameter_name, value=default_value)
                            else:
                                temp_dict = self.make_nested_dict(new_section[j], child_dict=temp_dict, parameter=None, value=None)
                        new_dict = temp_dict
                        for j in range(len(new_section)):
                            self.current_directory.append(new_section[j])
                    else:
                        new_dict = {new_section: {parameter_name: default_value}}
                        self.current_directory.append(new_section)
                    self.current_dict.update(new_dict)
                    new_section = None
                    self.update_current_dict()
                else:
                    if n == len(self.current_directory):
                        self.current_dict[parameter_name] = default_value
                    else:
                        while n < len(self.current_directory) and n != len(self.current_directory):
                            self.current_directory.pop(-1)
                        self.update_current_dict()
                        self.current_dict[parameter_name] = default_value
        
    
    def write_yaml(self, file):
        dict_to_yaml(file, self.M7_config_dict)