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
from contextlib import contextmanager
from click.testing import CliRunner
from mcmc import mcmc
from copy import deepcopy
from operator import itemgetter


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
        self.m.G1.clear()
        self.m.input_arg('./tests/test_input.txt')
        self.m.make_init_graph()
        self.assertIsInstance(self.m.G1,nx.Graph)
        self.assertTrue(nx.is_connected(self.m.G1),'Initial Graph is connected')#checks if initial graph is connected
        



    def test_graph_change_add(self):
        self.m.input_arg('./tests/test_input.txt')
        self.m.make_init_graph()
        self.assertEqual(self.m.graph_change(1,2),1)#Since no two non zero vertices are connected, passing two vertices other than vertex0 sould add an edge
        self.assertNotEqual(self.m.G1.number_of_edges(),self.m.G2.number_of_edges())#ensures that the new graph is not the same as the old graph
        self.m.G1.clear()
        self.m.G2.clear()


    def test_graph_change_bridge(self):
        self.m.input_arg('./tests/test_input.txt')
        self.m.make_init_graph()
        self.assertEqual(self.m.graph_change(0,2),-1)
        self.assertEqual(self.m.G1.number_of_edges(),self.m.G2.number_of_edges())
        self.m.G1.clear()
        self.m.G2.clear()


    def test_graph_change_remove(self):
        self.m.input_arg('./tests/test_input.txt')
        self.m.make_init_graph()
        v1=self.m.M[1]
        v2=self.m.M[2]
        wt=self.m.dist(v1,v2)
        print(wt)
        self.m.G1.add_edge(v1,v2,weight=wt)
        self.assertEqual(self.m.graph_change(1,2),0)
        self.assertNotEqual(self.m.G1.number_of_edges(),self.m.G2.number_of_edges())
        self.m.G1.clear()
        self.m.G2.clear()


    def test_calculate_bridges(self):
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
        
        deg,edge,path=self.m.main()
        
        #self.assertEqual(len(self.m.uniques),4)
        nodes=len(self.m.M)
        max_edges=nodes*(nodes-1)/2
        self.assertGreater(edge,nodes-1)#number of edges should be more than M-1 to stay connected. M is the number of nodes
        self.assertLessEqual(edge,max_edges)#number of edges should be less than or equal to M(M-1)/2 where M is the number of nodes
        self.assertGreater(deg,0)#to check that at least one or mode edges are connected to node 0
        self.assertLessEqual(deg,nodes-1)#deg of vertex 0 should not me more than M-1
        #self.m.G1.clear()
        #self.m.G2.clear()

    #Theta value should be greater than 0
    def test_theta_func(self):

        G=nx.star_graph(4)
        self.assertNotEqual(self.m.theta_func(G),0)

    def test_MH(self):
        self.m.input_arg('./tests/test_MH.txt')
        self.m.make_init_graph()
        self.m.G2=deepcopy(self.m.G1)
        v1=self.m.M[1]
        v2=self.m.M[2]
        self.m.G1.add_edge(v1,v2,weight=self.m.dist(v1,v2))
        print(self.m.G1.number_of_edges())
        print(self.m.G2.number_of_edges())
        self.assertEqual(self.m.MH(),1)
        self.m.G1.clear()
        self.m.G2.clear()

    #The input file has only three nodes. And the initial graph has two edges connecting node 0 to the other two nodes. The longer edge is of length 2
    def test_max_shortest_path(self):
        self.m.G1.clear()
        self.m.input_arg('./tests/test_MH.txt')
        self.m.make_init_graph()
        self.assertEqual(self.m.max_shortest_path(self.m.G1),2)
        
    #To check if graph counter actually keeps track of unique graph
    def test_graph_counter_behavior(self):
        self.m.input_arg('./tests/test_MH.txt')
        self.m.make_init_graph()
        self.assertEqual(len(self.m.uniques),0)
        self.m.graph_count(self.m.G1)
        self.assertEqual(len(self.m.uniques),1)
        flag=self.m.graph_change(1,2)
        self.m.graph_count(self.m.G2)
        self.assertEqual(len(self.m.uniques),2)
        self.m.graph_count(self.m.G1)
        self.assertEqual(len(self.m.uniques),2)
        key=frozenset(self.m.G1.edges(nbunch=self.m.M))
        self.assertEqual(self.m.uniques[key],2)
        self.m.uniques.clear()
        self.m.G1.clear()
        self.m.G2.clear()

    #Total count of graphs in the markov chain should be equal to the number of iterations. To check that.
    def test_total_graph_count(self):
        self.m.input_arg('./tests/test_input.txt')
        self.m.make_init_graph()
        self.m.mc_chain_generator()
        self.assertEqual(self.m.iterations,sum(self.m.uniques.values()))
        self.m.uniques.clear()
        
    #Our test_MH.txt has only three nodes hence there are only 4 different connected graphs possible. This checks that the number of uniques after simulation is not more than the maximum possible unique graphs
    def test_mc_chain_generator(self):
        self.m.input_arg('./tests/test_MH.txt')
        self.m.make_init_graph()
        self.m.mc_chain_generator()
        assert (len(self.m.uniques)>0 and len(self.m.uniques)<=4)
        self.m.G1.clear()
        self.m.G2.clear()
        
    #Testing the quantiling function when the number of unique graphs is less than 100. Since top 1% is fractional, the single most likely graph is returned 
    def test_quantiling_single(self):
        self.m.G1.clear()
        self.m.G2.clear() 
        self.m.input_arg('./tests/test_MH.txt')
        self.m.make_init_graph()
        self.m.mc_chain_generator()
        top=len(self.m.quantiling())
        self.assertEqual(top,1)
        #To check that the output file is not empty
        flag = os.path.exists(self.m.o_file)
        self.assertTrue(flag)

    
    #Testing the quantiling function when the number of unique graphs is greater than or equal 100
    def test_quantiling_multiple(self):
        self.m.G1.clear()
        self.m.G2.clear() 
        self.m.input_arg('./tests/test_input.txt')
        self.m.make_init_graph()
        self.m.mc_chain_generator()
        top=len(self.m.quantiling())
        print(top,(0.01*len(self.m.uniques)))
        assert (top-round(0.01*len(self.m.uniques)))<0.0001
        #To check that the output file is not empty
        flag = os.path.exists(self.m.o_file)
        self.assertTrue(flag)

