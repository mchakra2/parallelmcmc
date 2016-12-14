from operator import itemgetter
from copy import deepcopy
import math
import networkx as nx
import graphviz as gz
import numpy as np
import os
import timeit
import multiprocessing
import random
#from joblib import Parallel, delayed #For parallelization
from multiprocessing import Pool,Value
class MarkovChain:
    input_f='./IOFiles/input.txt'#default input file location
    o_file="./IOFiles/output.txt"#default output file location
    test=0
    G1=nx.Graph()
    G2=nx.Graph()
    iterations=200#Number of Steps in the simulation
    T=1
    r=1
    uniques={}#Empty dictionary to keep track of unique graphs and number of times they are observed
    exp_d0=0.0#Expectation of degree of vertex 0
    exp_edgs=0.0#Expectation of number of edges
    exp_max_path=0.0#Expection of the maximum value of minimum paths

    def main(self):
        start_time = timeit.default_timer()
        #Clear G1 and G2 in case they are not empty
        self.G1.clear()
        self.G2.clear()
        self.input_arg(self.input_f)
        self.make_init_graph()
        self.uniques.clear()
        self.exp_d0=0#Expectation of degree of vertex 0
        self.exp_edgs=0
        self.exp_max_path=0
        np = multiprocessing.cpu_count()
        print ('Number of CPUs to be used:  {0:1d}'.format(np))
        process_iter=[int(self.iterations/np) for i in range(np)]#Number of interations for each processes
        print(process_iter)
        #Creating the worker pool
        pool = Pool(processes=np)   
        # parallel map
        results=pool.map(self.mc_chain_generator, process_iter)#this is a list of tuples returned from the parallel operation
        for i in range(len(results)):
            self.exp_d0+=results[i][0]
            self.exp_max_path+=results[i][1]
            self.exp_edgs+=results[i][2]
            #unique_graphs
        print('test',self.test)
        #d0, E and avg_path are for calculating the expectations from the sum of these attributes over the period of simulation
        self.exp_d0=float(self.exp_d0)/self.iterations
        self.exp_edgs=float(self.exp_edgs)/self.iterations
        self.exp_max_path=float(self.exp_max_path)/self.iterations
        print('The expected number of edges connected to vertex 0 is ',self.exp_d0)
        print('The expected number of edges in the entire graph ',self.exp_edgs)
        print('The expected maximum distance of the shortest path in a graph that connects vertex 0 to another vertex',self.exp_max_path)
        print('The number of unique graphs ',len(self.uniques))
        #self.quantiling()
        elapsed = timeit.default_timer() - start_time
        print(elapsed)
        #return(self.exp_d0,E,avg_path)
    
   
    def dist(self,a,b):
        '''Function to calculate the weight of an edge. The weight is the Euclidian distance between two node tuples'''
        if type(a)==tuple and type(b)==tuple:
            wt=math.sqrt((a[1]-b[1])**2+(a[0]-b[0])**2)
            return(wt)

        else:
            print ("Inputs should be tuples")
            raise TypeError

        
    def input_arg(self, in_file):
        '''Fuction for Reading the input file and storing the nodes as a list of tuples''' 
        self.M=[]#list for the node tuples
        if os.path.exists(in_file)!= True:
            print ("The input file path does not exist. Default values will be used")
            raise IOError
        f = open(in_file)

        for line in f:
            li=line.strip()
   
            if not li.startswith("#"):#Ignore the lines starting with '#'
                if "=" in li:
                        if  li.split("=")[0]=='T':
                            self.T=float(li.split("=")[1])
                            #print(self.T)
                        elif li.split("=")[0]=='r':
                            
                            self.r=float(li.split("=")[1])
                            print(self.r)
                        elif li.split("=")[0]=='iterations':
                            self.iterations=int(li.split("=")[1])
                            print(self.iterations)

                else:
                    tmp = line.split(",")
                    self.M.append((float(tmp[0]), float(tmp[1])))
        
        f.close()

        
    
    def make_init_graph(self):
        '''Function to  make the initial graph G1 with the given  nodes. 
        I have just connected node 0 or the first node in M to all other nodes in M'''
        G1=nx.Graph()
        G1.add_nodes_from(self.M)
        for i in range(1,len(self.M)):
            
            G1.add_edge(self.M[0],self.M[i],weight=self.dist(self.M[0],self.M[i]))
        return(G1)
       
   
    def graph_change(self,idx1,idx2,G1):
        '''Function to change the state of the edge between the given nodes. 
        If the edge is not present, it is added else it is removed if it is not a bridge'''

        G2=deepcopy(G1)
        
        v1=self.M[idx1]
        v2=self.M[idx2]
        
        if G2.has_edge(v1,v2)==False:#Edge between v1 and v2 is not present. Hence add the edge.
            
            G2.add_edge(v1,v2,weight=self.dist(v1,v2))
            return(G2)
        
        else:
            if len(nx.minimum_edge_cut(G2,v1,v2))==1:#Edge is present but it is a bridge. The graph is not altered 

                return(-1)
            
            else:
                G2.remove_edge(v1,v2)#Edge is removed if it is present in graph but not a bridge
                return(G2)

    
    def calculate_bridges(self,G):
        '''Function to calculate the number of bridges in a given graph'''
        if type(G)!=nx.Graph:
            print ("Argument passed to the function should be a Graph")
            raise TypeError
            
        b=0
        for i in G.edges():
    
            if len(nx.minimum_edge_cut(G,i[0],i[1]))==1:
                b+=1
        return(b)
    
    
    def calculate_q(self,G):
        '''Function to calculate the q(m|n)  probabilitie by taking in Xn as argument'''
        b=self.calculate_bridges(G)
        nodes=float(G.number_of_nodes())
        q=(nodes*(nodes-1)/2)-b
        return(1/q)

    
    def theta_func(self,G):
        '''Funtion to return theta(Xi)'''
        theta=self.r*G.size(weight='weight')
        for i in range(1,G.number_of_nodes()):
            theta+=nx.shortest_path_length(G,source=G.nodes()[0],target=G.nodes()[i],weight='weight')

        return(theta)

    #Function to implement metropolis-hastings algorithm
    def MH(self,G1,G2):

        f=math.exp(-float(self.theta_func(G2)-self.theta_func(G1))/self.T)

        q=self.calculate_q(G2)/self.calculate_q(G1)
        aij=min(f*q,1)
        U=np.random.random()#randomly chosen number between 0 and 1
        if aij>=U: 
            return(1)#flag to accept the proposed graph change if aij >= U
        else:
            return(0)

    #Function to return the maximum of the shortest path from vertex 0 to other vertices
    def max_shortest_path(self,G):
        P=nx.shortest_path_length(G, source=self.M[0],weight='weight')
        max_path=-1
        for i in P:
            max_path=max(max_path,P[i])
        #P=Parallel(n_jobs=2)(delayed(nx.shortest_path_length)(G,source=self.M[0],target=i,weight='weight') for i in self.M[1:])
        #max_path=np.amax(P)
        return(max_path)

    #Function to count uniques
    def graph_count(self,G,uniques):
        key=frozenset(G.edges(nbunch=self.M))

        if key in uniques:#increment count if G has been observed before
            uniques[key]+=1
        else:#add to the dictionary if G has not been observed before
            uniques[key]=1
            

    #Function to generate the markov chain
    def mc_chain_generator(self,iterations):
        unique_graphs={}
        #uniques.clear()
        self.test+=1
        exp_d0=0#Expectation of degree of vertex 0
        exp_edgs=0
        exp_max_path=0
        G1=self.make_init_graph()#initial graph
        for i in range(iterations):#Propose graph  modification at each simulation step
            G2=-1
            while(G2==-1):#if  the randomly selected edge is a bridge select a different edge
                A=random.sample(range(len(self.M)), 2)#Choose a tuple randomly without replacement from the range of indices in M
                #A=np.random.choice(len(self.M), 2,replace=0)#Choose a tuple randomly without replacement from the range of indices in M
                G2=self.graph_change(A[0],A[1],G1)#graph_change  function returns -1 if the edge  is  a bridge 

                            
            accept=self.MH(G1,G2)
            if accept==1:#Accept the change if MH function returns 1
                #print(i,"accept")
                G1=deepcopy(G2)

            #Maintaining running averages for performing statistical analysis later
            exp_d0+=G1.degree(self.M[0])
            exp_edgs+=G1.number_of_edges()
            exp_max_path+=self.max_shortest_path(G1)
            self.graph_count(G1,unique_graphs)
        
        return(exp_d0,exp_max_path,exp_edgs,unique_graphs)
        


    def quantiling(self,dictionary):
        '''Function to take in a dictionary of unique graphs and their occurances.
        It returns a list of edge-list of top 1% graphs generated in the markov chain'''
        print('The edge lists of the top 1% graphs are printed in ', self.o_file)
        f=open(self.o_file,"w")#Writing into the output file
        
        desc_adj=sorted(dictionary.items(), key=itemgetter(1), reverse=True)
        top=0.01*len(dictionary)
        top_graphs=[]
        if top<1:
            print('Since there are less than 100 unique graphs, only the most likely graph will be written in the output file')
            top_graphs.append(desc_adj[0][0])
            f.write("%s\n"%top_graphs[0])
        else:
            i=0
            while(i<=round(top,0)-1):
                top_graphs.append(desc_adj[i][0])
                f.write("%s\n"%top_graphs[i])
                i+=1
        
        f.close()
        return(top_graphs)

if __name__ == '__main__':
    m=MarkovChain()
    m.main()



   
