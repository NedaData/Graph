import numpy as np
from scipy import linalg
import json

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adjacentlist = []
    
        i = 0

        while (i < self.vertices):
            self.adjacentlist.append([])
            i+=1
    
    def read_json_file(filename):
        with open(filename) as f:
            js = json.load(f)
        return js



    def addLinks(self, source, target):
        if(source > self.vertices or source < 0 
           or target > self.vertices or target < 0):
            return
        
        self.adjacentlist[source].append(target)
        self.adjacentlist[target].append(source)

    def countEdges(self):
        Sum = 0
        for i in range(self.vertices):
            Sum += len(self.len(self.adjacentlist[i]))
        return Sum // 2
    
    def countVertices(self):
        Sum = 0
        for _ in range(self.vertices):
            Sum += 1
        return Sum
    
    def degree(Graph, vertex):
        degree = 0
        for i in range(Graph.vertices):
            if Graph[vertex][i] == 1:
                degree +=1
        return degree
    
    

    def dfs(self, visited, start):
        if(start < 0 or start > self.vertices):
            return
        
        visited[start] = True
        i = 0
        while(i < len(self.adjacentlist[start])):
            if(visited[self.adjacentlist[start][i]] == False):
                self.dfs(visited, self.adjacentlist[start][i])
            
            i+=1
    
    def countSubGraph(self):
        visited = [False] * (self.vertices)
        result = 0
        i = 0
        while(i < self.vertices):
            visited[i] = False
            i+=1
        
        i = 0
        while(i < self.vertices):
            if(visited[i] == False):
                result+=1
                self.dfs(visited, i)
            
            i+=1
        
    def isConnected(self):
        vertices = self.vertices
        
        visited = [vertices]
        
        self.dfs(visited, 0)
        count = 0
        for i in range(len(visited)):
            if(visited[i]):
                count+=1
        if (vertices == count):
            return ("Graph is connected")
        else:
            return ("Graph is not connected")
        
    def dfsCC(self, temp, source, visited):
        visited[source] = True

        temp.append(source)

        for i in self.adjacentlist[source]:
            if visited[i] == False:

                temp = self.dfsCC(temp, i, visited)
        
        return temp
    
    def isolateVertex(self):
        count = 0
        visited = []
        cc = []
        for i in range(self.vertices):
            visited.append(False)

        for source in range(self.vertices):
            if visited[source] == False:
                temp = []
                cc.append(self.dfsCC(temp, source, visited))
                count +=1
        
        return count, cc
    
    def convertAdjLIstToAdjMatrix(self):
        matrix = [[0 for j in range(self.vertices)]
                    for i in range(self.vertices)]
        
        for i in range(self.vertices):
            for j in self.adjacentlist[i]:
                matrix[i][j] = 1
        
        return matrix

    def eigenVectorCentrality(self):
        AdjMatrix = self.convertAdjLIstToAdjMatrix()
        n = len(AdjMatrix)
        vals, vecs = linalg.eig(AdjMatrix)
        i = np.argmax(vals)
        return np.abs(vecs[:, i])


    def betweeness_centrality(self):
        AdjMatrix = self.convertAdjLIstToAdjMatrix() 
        betweenness = dict.fromkeys(AdjMatrix, 0.0)

        nodes = AdjMatrix.nodes
        for s in nodes:
            Stack = []
            predessor = {}

            for v in AdjMatrix:
                predessor[v] = []
            
            sigma_val = dict.fromkeys(AdjMatrix, 0.0)

            Dist = {}
            sigma_val[s] = 1.0

            Dist[s] = 0
            Queue = [s]

            #BFS algorithm
            while Queue:

                v = Queue.pop(0)
                Stack.append(v)
                #distance of v
                d = Dist[v]
                sigmav = sigma_val[v]

                for g in AdjMatrix[v]:
                    if g not in Dist:
                        Queue.append(g)
                        Dist[g] =d+1
                    if Dist[g] == d+1:
                        sigma_val[g] +=sigmav
                        predessor[g].append(v)
            
            delta = dict.fromkeys(Stack, 0)
            while Stack:
                g = Stack.pop()
                magic = (1.0 + delta[g]) / sigma_val[g]
                
                for v in predessor[g]:
                    delta[v] += sigma_val[v] * magic
                
                if g != s:
                    betweenness[g] += delta[g]

        scale = 0.5
        for b in betweenness:
            betweenness[b] *= scale

        b_centrality = list(betweenness.values)
        return b_centrality

    def pageRank(self, AdjacentMatrix, d, falff = None):
        """
        falff - Nx1 np.ndarray (Initial page rank probability)
        d - dumping factor, most commonly 0.85
        """
        
        N = len(AdjacentMatrix)
        if falff is None:
            norm_falff  = np.ones((N, )) / N
        else:
            norm_falff = falff / np.sum(falff)
        
        deg = np.sum(AdjacentMatrix, axis = 0)
        deg[deg == 0] = 1
        D1 = np.diag(1 / deg)
        B = np.eye(N) - d* np.dot(AdjacentMatrix, D1)
        b = (1-d)*norm_falff
        r = linalg.solve(B, b)
        r /= np.sum(r)
        return r
    
    def computeRank(self):
        rankList = []
        adjacentMatrix = self.convertAdjLIstToAdjMatrix()
        rank = self.pageRank(adjacentMatrix, 0.85)
        rankList = sorted(rank, key=lambda x: x[1], reverse=True)
        return rankList
    
    def main(self):
        file = "../graphSolution/challenge_graph.json"
        links = self.read_json_file(file)["links"]
        nodes = self.read_json_file(file)["nodes"]
        num_vert = self.countVertices(nodes)
        G = Graph(num_vert)
        for i in range(len(links)):
            G.addLinks(int(links[i]["source"]), int(links[i]["target"]))
        #Apply other functions
        return G

    

    if __name__ == "__main__":
        main()

    









        
        
   


        


