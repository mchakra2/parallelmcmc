from operator import itemgetter
from copy import deepcopy
import math
import networkx as nx
import graphviz as gz
import numpy as np
import os
import timeit
import joblib

class MarkovChain:
    input_f='./IOFiles/input.txt'#default input file location
    o_file="./IOFiles/output.txt"#default output file location

    G1=nx.Graph()
    G2=nx.Graph()
    iterations=200#Number of Steps in the simulation
    T=1
    r=1
    uniques={}#Empty dictionary to keep track of unique graphs and number of times they are observed
    
    def main(self):
        start_time = timeit.default_timer()
        #Clear G1 and G2 in case they are not empty
        self.G1.clear()
        self.G2.clear()
        self.input_arg(self.input_f)
        self.make_init_graph()
        self.uniques.clear()
        self.mc_chain_generator()
        #d0, E and avg_path are for calculating the expectations from the sum of these attributes over the period of simulation
        d0=float(self.exp_d0)/self.iterations
        E=float(self.exp_edgs)/self.iterations
        avg_path=float(self.exp_max_path)/self.iterations
        print('The expected number of edges connected to vertex 0 is ',d0)
        print('The expected number of edges in the entire graph ',E)
        print('The expected maximum distance of the shortest path in a graph that connects vertex 0 to another vertex',avg_path)
        print('The number of unique graphs ',len(self.uniques))
        self.quantiling()
        elapsed = timeit.default_timer() - start_time
        print(elapsed)
        return(d0,E,avg_path)
    
    '''Function to calculate the weight of an edge. The weight is the Euclidian distance between two node tuples'''
    def dist(self,a,b):
        if type(a)==tuple and type(b)==tuple:
            wt=math.sqrt((a[1]-b[1])**2+(a[0]-b[0])**2)
            return(wt)

        else:
            print ("Inputs should be tuples")
            raise TypeError

    '''Fuction for Reading the input file and storing the nodes as a list of tuples'''     
    def input_arg(self, in_file):
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

        
    '''Function to  make the initial graph G1 with the given  nodes. I have just connected node 0 or the first node in M to all other nodes in M'''
    def make_init_graph(self):
        
        self.G1.add_nodes_from(self.M)
        for i in range(1,len(self.M)):
            
            self.G1.add_edge(self.M[0],self.M[i],weight=self.dist(self.M[0],self.M[i]))
       
    '''Function to change the state of the edge between the given nodes. If the edge is not present, it is added else it is removed if it is not a bridge'''
    def graph_change(self,idx1,idx2):
        self.G2=deepcopy(self.G1)
        
        v1=self.M[idx1]
        v2=self.M[idx2]
        
        if self.G2.has_edge(v1,v2)==False:#Edge between v1 and v2 is not present. Hence add the edge.
            
            self.G2.add_edge(v1,v2,weight=self.dist(v1,v2))
            return(1)
        
        else:
            if len(nx.minimum_edge_cut(self.G2,v1,v2))==1:#Edge is present but it is a bridge. The graph is not altered 

                return(-1)
            
            else:
                self.G2.remove_edge(v1,v2)#Edge is removed if it is present in graph but not a bridge
                return(0)

    #Function to calculate the number of bridges in a given graph
    def calculate_bridges(self,G):
        if type(G)!=nx.Graph:
            print ("Argument passed to the function should be a Graph")
            raise TypeError
            
        b=0
        for i in G.edges():
    
            if len(nx.minimum_edge_cut(G,i[0],i[1]))==1:
                b+=1
        return(b)
    
    #Function to calculate the q(m|n)  probabilitie by taking in Xn as argument
    def calculate_q(self,G):
        b=self.calculate_bridges(G)
        nodes=float(G.number_of_nodes())
        q=(nodes*(nodes-1)/2)-b
        return(1/q)

    #Funtion to return theta(Xi)
    def theta_func(self,G):
        theta=self.r*G.size(weight='weight')
        for i in range(1,G.number_of_nodes()):
            theta+=nx.shortest_path_length(G,source=G.nodes()[0],target=G.nodes()[i],weight='weight')

        return(theta)

    #Function to implement metropolis-hastings algorithm
    def MH(self):

        f=math.exp(-float(self.theta_func(self.G2)-self.theta_func(self.G1))/self.T)

        q=self.calculate_q(self.G2)/self.calculate_q(self.G1)
        aij=min(f*q,1)
        U=np.random.random()#randomly chosen number between 0 and 1
        if aij>=U: 
            return(1)#flag to accept the proposed graph change if aij >= U
        else:
            return(0)

    #Function to return the maximum of the shortest path from vertex 0 to other vertices
    def max_shortest_path(self,G):
        #start_time = timeit.default_timer()
        P=nx.shortest_path_length(G, source=self.M[0],weight='weight')
        max_path=-1
        for i in P:
            max_path=max(max_path,P[i])
        #elapsed = timeit.default_timer() - start_time
        #print(elapsed)
        return(max_path)

    #Function to count uniques
    def graph_count(self,G):
        key=frozenset(G.edges(nbunch=self.M))

        if key in self.uniques:#increment count if G has been observed before
            self.uniques[key]+=1
        else:#add to the dictionary if G has not been observed before
            self.uniques[key]=1

    #Function to generate the markov chain
    def mc_chain_generator(self):
        diff_g=0
        self.uniques.clear()
        self.exp_d0=0#Expectation of degree of vertex 0
        self.exp_edgs=0
        self.exp_max_path=0
        for i in range(self.iterations):#Propose graph  modification at each simulation step
            flag=-1
            while(flag==-1):#if  the randomly selected edge is a bridge select a different edge

                A=np.random.choice(len(self.M), 2,replace=0)#Choose a tuple randomly without replacement from the range of indices in M
                flag=self.graph_change(A[0],A[1])#graph_change  function returns -1 if the edge  is  a bridge 

                            
            accept=self.MH()
            if accept==1:#Accept the change if MH function returns 1
                
                self.G1=deepcopy(self.G2)

            #Maintaining running averages for performing statistical analysis later
            self.exp_d0+=self.G1.degree(self.M[0])
            self.exp_edgs+=self.G1.number_of_edges()
            self.exp_max_path+=self.max_shortest_path(self.G1)
            self.graph_count(self.G1)
        self.G1.clear()
        self.G2.clear()
        print(len(self.uniques))

    #Function to return a list of edge-list of top 1% graphs generated in the markov chain
    def quantiling(self):
        print('The edge lists of the top 1% graphs are printed in ', self.o_file)
        f=open(self.o_file,"w")#Writing into the output file
        
        desc_adj=sorted(self.uniques.items(), key=itemgetter(1), reverse=True)
        top=0.01*len(self.uniques)
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
