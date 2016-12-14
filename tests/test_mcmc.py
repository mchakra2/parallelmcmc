#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mcmc
----------------------------------

Tests for `mcmc` module.
"""

import math
import sys
import os
import unittest
import networkx as nx
import random
from contextlib import contextmanager
from click.testing import CliRunner
from mcmc import mcmc
from copy import deepcopy
from operator import itemgetter
from multiprocessing import Pool
import multiprocessing
class TestMcmc(unittest.TestCase):

    def setUp(self):
        #pass
        self.m=mcmc.MarkovChain()

    def tearDown(self):
        pass

    def test_dist_arg_type(self):#test if the dist function rises errors when given non numerical tuples  as input
       
        with self.assertRaises(TypeError):
            self.m.dist((-4,1),1)
        with self.assertRaises(TypeError):
            self.m.dist(2.2,'a')
        with self.assertRaises(TypeError):
            self.m.dist(('A',3),(2.2,1))



    def test_dist_return(self):#To check if dist function returns the cartesian distance between  given tuples   
        
        val=round(math.sqrt(2.2**2+2.2**2),2)
        wt=self.m.dist((2.2,2.2), (0,0))
        self.assertEqual(round(wt,2),val)
        self.assertEqual(self.m.dist((2,0),(2,0)),0)
    
    def test_input_file(self):#Test that input_arg raises error if input file does not exist and is not  empty
           
        self.assertRaises(IOError,self.m.input_arg,'random_file_which_should_not_exist.txt')
        #self.m.main()
        flag = os.path.getsize(self.m.input_f)
        self.assertGreater(flag,0)
        self.assertGreater(self.m.T,0)#makes sure that T is strictly positive

    def test_tuples(self):
        self.m.input_arg('./tests/test_input.txt')
        self.assertGreater(len(self.m.M),0)#To ensure that the number of vertices is greater than 0

    def test_make_init_graph(self):
        #self.m.G1.clear()
        self.m.input_arg('./tests/test_input.txt')
        G1=self.m.make_init_graph()
        self.assertIsInstance(G1,nx.Graph)
        self.assertTrue(nx.is_connected(G1),'Initial Graph is connected')#checks if initial graph is connected
        


    
    def test_graph_change_add(self):
        '''Checks if graph_change function adds an edge 
        if it is not already present between the selected nodes'''
        self.m.input_arg('./tests/test_input.txt')
        G1=self.m.make_init_graph()
        G2=self.m.graph_change(1,2,G1)
        flag=G2.number_of_edges()-G1.number_of_edges()
        self.assertEqual(flag,1)#Since no two non zero vertices are connected, passing two vertices other than vertex0 sould add an edge
        self.assertTrue(nx.is_connected(G2))#ensures that the new graph is connected
        #self.m.G1.clear()
        #self.m.G2.clear()


    def test_graph_change_bridge(self):
        '''Checks if graph_change function keeps the graph
        unchanged a bridge is present between the selected nodes'''
        self.m.input_arg('./tests/test_input.txt')
        G1=self.m.make_init_graph()
        self.assertEqual(self.m.graph_change(0,2,G1),-1)
        #self.assertEqual(self.m.G1.number_of_edges(),self.m.G2.number_of_edges())
        self.m.G1.clear()
        #self.m.G2.clear()


    def test_graph_change_remove(self):
        '''Checks if graph_change function the edge 
        present between the selected nodes provided it is not a bridge'''
        self.m.input_arg('./tests/test_input.txt')
        G1=self.m.make_init_graph()
        v1=self.m.M[1]
        v2=self.m.M[2]
        wt=self.m.dist(v1,v2)
        #print(wt)
        G1.add_edge(v1,v2,weight=wt)
        G2=self.m.graph_change(1,2,G1)
        flag=G2.number_of_edges()-G1.number_of_edges()
        self.assertEqual(flag,-1)#to ensure that an edge has been removed
        self.assertTrue(nx.is_connected(G2))#ensures that the new graph is connected
        #self.m.G1.clear()
        #self.m.G2.clear()


    def test_calculate_bridges(self):
        '''This tests if the calculate_bridges function
        can calculate the number of bridges in a provided 
        graph correctly'''
        a=1
        self.assertRaises(TypeError,self.m.calculate_bridges,a)
        G=nx.cycle_graph(3)
        self.assertEqual(0,self.m.calculate_bridges(G))
        G1=nx.star_graph(5)
        self.assertEqual(5,self.m.calculate_bridges(G1))
        G1.clear()
        G.clear()
        
    def test_calculate_q(self):
        G1=nx.star_graph(5)
        self.assertGreater(self.m.calculate_q(G1),0)
        self.assertLess(self.m.calculate_q(G1),1)
        self.assertEqual(round(self.m.calculate_q(G1),2),0.1)
        G1.clear()

    def test_main(self):
        '''Tests the sanity of the return values of the main function'''
        self.m.main()
        #self.assertEqual(len(self.m.uniques),4)
        nodes=len(self.m.M)
        max_edges=nodes*(nodes-1)/2
        self.assertGreater(self.m.exp_edgs,nodes-1)#number of edges should be more than M-1 to stay connected. M is the number of nodes
        self.assertLessEqual(self.m.exp_edgs,max_edges)#number of edges should be less than or equal to M(M-1)/2 where M is the number of nodes
        self.assertGreater(self.m.exp_d0,0)#to check that at least one or more edges are connected to node 0
        self.assertLessEqual(self.m.exp_d0,nodes-1)#deg of vertex 0 should not me more than M-1
        #To check that the output file is not empty
        flag = os.path.exists(self.m.o_file)
        self.assertTrue(flag)

        #self.m.G1.clear()
        #self.m.G2.clear()


    #Theta value should be greater than 0
    def test_theta_func(self):

        G=nx.star_graph(4)
        self.assertNotEqual(self.m.theta_func(G),0)

    def test_MH(self):
        self.m.input_arg('./tests/test_MH.txt')
        G1=self.m.make_init_graph()
        G2=deepcopy(G1)
        v1=self.m.M[1]
        v2=self.m.M[2]
        G1.add_edge(v1,v2,weight=self.m.dist(v1,v2))
        #print(self.m.G1.number_of_edges())
        #print(G2.number_of_edges())
        self.assertEqual(self.m.MH(G1,G2),1)
        #self.m.G1.clear()
        #self.m.G2.clear()

    #The input file has only three nodes. And the initial graph has two edges connecting node 0 to the other two nodes. The longer edge is of length 2
    def test_max_shortest_path(self):
        #self.m.G1.clear()
        self.m.input_arg('./tests/test_MH.txt')
        G1=self.m.make_init_graph()
        self.assertEqual(self.m.max_shortest_path(G1),2)
        
    #To check if graph counter actually keeps track of unique graph
    def test_graph_counter_behavior(self):
        self.m.input_arg('./tests/test_MH.txt')
        G1=self.m.make_init_graph()
        uniques={}
        self.assertEqual(len(uniques),0)
        self.m.graph_count(G1,uniques)
        self.assertEqual(len(uniques),1)
        G2=self.m.graph_change(1,2,G1)
        self.m.graph_count(G2,uniques)
        self.assertEqual(len(uniques),2)
        self.m.graph_count(G1,uniques)
        self.assertEqual(len(uniques),2)
        key=frozenset(G1.edges(nbunch=self.m.M))
        self.assertEqual(uniques[key],2)
        self.m.uniques.clear()
        #self.m.G1.clear()
        #self.m.G2.clear()

    #Total count of graphs in the markov chain should be equal to the number of iterations. To check that.
    def test_total_graph_count(self):
        self.m.input_arg('./tests/test_input.txt')
        self.m.make_init_graph()
        exp_d0,exp_edgs,exp_max_path,uniques=self.m.mc_chain_generator(self.m.iterations)
        self.m.mc_chain_generator(self.m.iterations)
        self.assertEqual(self.m.iterations,sum(uniques.values()))
        self.m.uniques.clear()
        
    #Our test_MH.txt has only three nodes hence there are only 4 different connected graphs possible. This checks that the number of uniques after simulation is not more than the maximum possible unique graphs
    def test_mc_chain_generator(self):
        self.m.input_arg('./tests/test_MH.txt')
        exp_d0,exp_edgs,exp_max_path,uniques=self.m.mc_chain_generator(self.m.iterations)
        assert (len(uniques)>0 and len(uniques)<=4)
        #self.m.G1.clear()
        #self.m.G2.clear()
        
    #Testing the quantiling function when the number of unique graphs is less than 100. Since top 1% is fractional, the single most likely graph is returned 
    def test_quantiling_single(self):
        self.m.input_arg('./tests/test_MH.txt')
        exp_d0,exp_edgs,exp_max_path,uniques=self.m.mc_chain_generator(self.m.iterations)
        top=len(self.m.quantiling(uniques))
        self.assertEqual(top,1)
        #To check that the output file is not empty
        flag = os.path.exists(self.m.o_file)
        self.assertTrue(flag)

    
    #Testing the quantiling function when the number of unique graphs is greater than or equal 100
    def test_quantiling_multiple(self):
        self.m.input_arg('./tests/test_input.txt')
        exp_d0,exp_edgs,exp_max_path,uniques=self.m.mc_chain_generator(self.m.iterations)
        top=len(self.m.quantiling(uniques))
        print(top,(0.01*len(uniques)))
        assert (top-round(0.01*len(uniques)))<0.0001
        #To check that the output file is not empty
        flag = os.path.exists(self.m.o_file)
        self.assertTrue(flag)

