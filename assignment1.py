

# ==========
# Q1

def calculate_cuteness(left, right):
    """
    This function calculate the 'cuteness' value from two fused fitmons.

    Inputs:
    left (list): The fitmon on the left, where left[1] is the cuteness and left[2] is the right attribute.
    right (list): The fitmon on the right, where right[1] is the cuteness and right[0] is the left attribute.

    Returns:
    int with the calculated cuteness value.
    """
    return int((left[1] * left[2]) + (right[1] * right[0]))




def fuse(fitmons):
    """
    This function does  is fuse  a list of fitmons
    where each fitmon have three values are [affinity_left, cuteness_score, affinity_right].
    This function will perform the best optimization to merge these value in a specific way to maximize a "cuteness" value.
    Written by Edison Len Hao Jie

   Mindset Approach : Since this  function calculates the maximum cuteness score achievable by optimally fusing
    `fitmons`. Each `fitmon` is represented as a three value list  [L, C, R], and when two such list
    are fused, a new cuteness score is generated as C1*R1 + C2*L2. So when i was thinking on how to approach this question
    , i draw out square brackets  represent two list and then imagine how the fusion goes  with conditions set for assignment in mind.
    While doing so i realize that these two square brackets i have drawn represents the lists looks like two matrice doing multiplication , by having that in mind
    i search matrix multiplication stuff relating to coding and found matrix chain multiplication . As it starts with list
    of matrices (like fitmons ) and slowly combine them in one matrix (maximum cuteness) but in the most optimal way . When starting we calculate
    the first two matrice, we can call it a first subproblem as we are going to use the knowledge of this calucation on more and bigger problem
    onwards like doing three matrices ,four five to n multiplications. But firstly , we are going find all the possible ways of  how the smalllest subproblem
    starts like maybe AB ,CD ,DE, CE .... if we Have A B C D E matrices.Afterwards we slowly build up larger subproblem until the max
    number of matrices given by using calculation we previously have done . In the end , the most optimal calculated matrix is shown in the end .
    This basically sound just like what our assignment doing . Additionally, this approach idea looks similar to how  karatsuba algorithm do its calculation as well .


    Input:
        fitmons where they are list of lists where each inner list contains three values.
    Return:
        Maximum cuteness_score from fusing.

    Time complexity:

        Best case analysis:O(n^3) , where n is the number of fitmons
        The function employs three nested loops corresponding to the length of sub-problems,
         starting indices for these sub-problems, and possible split points within these sub-problems.
         Additionally, each loop in the nested loops is O(n).
         Therefore, makes n^3

        Worst case analysis:O(n^3), Same explaination for best case .

    Space complexity:
        Input space analysis:O(n), the space needed to store the input list fitmons which is n.
        Aux space analysis::O(n^2), the space required for the FusingChamber table which is n times n that  store  the results of subproblems.

    """

    # number of fitmons which are lists store in n
    n = len(fitmons)
    # intializes a list named Fusing Chamber  of n times n and each cell in the matrix are initially set to 0.
    FusingChamber = [[0] * n for _ in range(n)]
    # this loop iterates over each fitmon using the index i
    for i in range(n):
    # set the FusingChamber to the fitmon by index i which mean the value of cuteness of the fitmon is just by itself
        FusingChamber[i][i] = fitmons[i]
    # this loop iterates over possible difference between sublists of fitmons ,
    # starting from the smallest gap difference of 1  between two fitmons up to n-1  that is biggest gap from start to end
    for gap in range(1, n):
        # this nested loop determines the starting index start in each gap size
        for start in range(n - gap):
            # this calculates the ending index end of the sublist that starts at index start and the gap
            end = start + gap
            # intializes a variable current_best_cuteness to zero ,
            # which will be used to track the maximum cuteness value found for the sublist from index start to end
            current_best_cuteness = 0
            # this loop iterates over index split from end - 1 ,
            # splitting the sublist into two parts at split to try different combinations of fusing fitmons
            for split in range(start, end):
                # left and right hold the best results of fusing the fitmons from start to split
                # and from split + 1 to end , respectively , based on previous calculations stored in FusingChamber
                left = FusingChamber[start][split]
                right = FusingChamber[split + 1][end]
                # calculates the cuteness of the current fusion by using the function calculate_cuteness
                # , combining the results of fusing left and right parts
                cuteness = calculate_cuteness(left, right)
                # creating a new list called fusion that show the result of fusion attributes from start to end
                fusion = [left[0], cuteness, right[2]]
                # this conditional checks if the newly computed cuteness is greater
                # than the current_best_cuteness for the sublist from start to end
                if cuteness > current_best_cuteness:
                    #if the new cuteness is higher , update current_best_cuteness and set FusingChamber[start][end]
                    #to the fused fitmon, storing the best result for this range
                    current_best_cuteness = cuteness
                    FusingChamber[start][end] = fusion
    # the function returns the maximum cuteness value from the entire fusion list of fitmons stored in FusingChamber[0][n-1][1]
    return FusingChamber[0][n - 1][1]
# ==========
# Q2

import heapq

class Vertex:
    """
           This class has these fuctions inside it that intialize the vertices and its attributes and conditions
           Written by Edison Len Hao Jie

           Mindset Approach: creating the vertices needed and the attributes . The first function creates attributes that  are the id of vertices , checking if the vertex
           has been visited or not , the shortest distance between vertices , locating the previous vertex , checking if the exit is found or not and lastly
           the edges created from the vertices. The second function just add the edges to self.edges. The third function and fourth function are just comparing shortest distances ,
            are they lesser than or greater equal bewteen each other.The last function is for self use when  test casing and checking for error in the vertices .

           Input:
               None
           Return:
               None


           Time complexity:
               Best case analysis:O(1) because all the function in this class operate in constant time , O(1)
                                  because they perform a fixed amount of operations.
               Worst case analysis:O(1),same explanation for best case
           Space complexity:
               Input space analysis:O(1)  because all the function in this class use  constant space, O(1)
                                  because they use a fixed amount of space.
               Aux space analysis: O(1), because the space required to create vertices is constant
           """
    def __init__(self, id):
        # the id of the vertices
        self.id = id
        # checking if the vertex has been visited or not
        self.visited = False
        # the shortest distance betwwen vertices
        self.shortest_distance = float('inf')
        # locating the previous vertex
        self.previous_vertex = None
        # checking if the exit is found or not
        self.exit = False
        # the edges from vertices
        self.edges = []

    def add_edge(self, edge):
        #if have edges add them into self.edges
        self.edges.append(edge)

    def __lt__(self, other):
        # checking if the new shortest distance is lesser than the other shortest distance
        return self.shortest_distance < other.shortest_distance

    def __ge__(self, other):
        # checking if the new shortest distance is  greater or equal than the other shortest distance
        return self.shortest_distance >= other.shortest_distance

    def __repr__(self):
        # use for test casing to track the vertices and edges being used
        return f"Vertex ID = {self.id}, Edges = {str(self.edges)}"

class Edge:
    """
               This class has these fuctions inside it  that intialize the vertices and its attributes and conditions
               Written by Edison Len Hao Jie

               Mindset Approach: creating the edges needed and the attributes . The first function creates attributes that are the edges have a start , end and weight used to go from one to another.
               The second function is just for self use when test casing and checking for error in the edges .



               All functions in this class has these time and space complexities .
               Time complexity:
                   Best case analysis:O(1),because all the function in this class operate in constant time , O(1)
                                  because they perform a fixed amount of operations.
                   Worst case analysis:O(1),same explanation for best case
               Space complexity:
                   Input space analysis:O(1) ,because all the function in this class use  constant space, O(1)
                                  because they use a fixed amount of space.
                   Aux space analysis:O(1) ,because the space required to create vertices is constant
               """
    def __init__(self, start, end, weight=1):
        #the start of the edge
        self.start = start
        #the end of the edge
        self.end = end
        #the weight used to cross one vertex to another
        self.weight = weight

    def __repr__(self):
        # for self use when test casing to track the path when finding the shortest distance
        return f"{self.start.id} -- {self.weight} --> {self.end.id}"

class TreeMap:
    """
    this class here is to intialize the graph to put vertices and edges in .
    """

    def __init__(self, roads, solulus):
        """
        This init function is used to intialize the graph
        Written by Edison Len Hao Jie

        Mindset approach:  Since the treemap is a graph and i have created the vertices(trees) and edges (roads).Now i shall input how many vertices and edges into the treemap.
        Since we have a condition that we must have destroy a solulu to escape. However, when destroying a solulu tree , we will teleport to a random tree.
         Therefore, i have came up the multiverse idea by creating two graphs and combining them to find the shortest distance from start to end.
         The first graph will contain the treemap specified in the input except we will not have an exit node. The second graph will be the same as the first graph but this will have the exit nodes and no start nodes.
         These 2 graphs  are stored in the same adjacency list where the index of graph 2  = index of graph 1 + N. Additionally,
         These 2 graphs are connected via the solulu trees with an edge from the solulu tree on first graph, to the vertex on the second graph.
         For example, if breaking a solulu tree teleports you to tree number 5, the edges would connect from the solulu tree in graph 1, to tree number 5 in graph 2.
          These edges we add only move from the 1st graph to the 2nd graph, and it will have the edge weight of the time it takes to chop down the solulu tree.
          Therefore, traversing this edge can be seen as chopping the solulu tree. Once we contructed this layered graph structure, we use dijkstra to find the shortest path from the start to the exit.
           The shortest path taken are depended on how many weight (w) is accumulate from roads taken between tree and
        time taken to destroy a solulu tree and teleport .

        Input:

              roads :roads are list of tuples that contains the (u,v,w) where u is the vertex(tree), v is the edge(road) and w is the weight taken to travel from one vertex to another.

             solulus: solulus are list of tuples that contains the (u,w,v) where u is the vertex(tree) , w is the time taken to destroy the solulu tree , v is the random vertex(tree) you will teleport to .
        Return:
            the treemap(graph) with the roads (edges ) and the trees and solulu trees(vertices)

        Time complexity:
            Best case analysis:O(|T| + |R|) ,where T is the number of vertices and R is the number of edges,
                               since we iterate over each list of inputs once.
            Worst case analysis:O(|T| + |R|) same as explanation as base case .
        Space complexity:
            Input space analysis:O(|T| + |R|) , the space required to store the T which are the number of verticles and R which are the number of edges .
            Aux space analysis:O(|T| + |R|) , the space required to create the graph with T which are the number of verticles and R which  are the number of edges .
        """
        # Initialize the maximum tree index based on the provided roads
        self.N = 0
        # Check each tree in the road tuples to update the maximum index
        for (u, v, w) in roads:

            if u + 1 > self.N:
                self.N = u + 1

            if v + 1 > self.N:
                self.N = v + 1
        # Create vertices list for the graph; doubling the amount to represent the second graph vertices as well
        self.graph = [Vertex(ID % self.N) for ID in range(2 * self.N)]

        # Add edges to the graph based on the roads input
        for (u, v, w) in roads:
            # Create a new edge in the first graph
            new_edge = Edge(self.graph[u], self.graph[v], w)
            self.graph[u].add_edge(new_edge)
            # Create a corresponding edge in the second graph for teleportation handling
            new_edge = Edge(self.graph[u + self.N], self.graph[v + self.N], w)
            self.graph[u + self.N].add_edge(new_edge)
        # Add teleportation edges based on the solulus input
        for (u, w, v) in solulus:
            # These edges connect a vertex in the first graph to a corresponding vertex in the second graph
            new_edge = Edge(self.graph[u], self.graph[v + self.N], w)
            self.graph[u].add_edge(new_edge)

    def escape(self, start, exits):
        """
        This function is used for input the start and exits in the treemap and show the shortest path to escape form start to exit .
        Written by Edison

        Mindset approach:  Since the treemap is a graph and i have created the vertices(trees) and edges (roads).Now i shall input how many vertices and edges into the treemap.
        Since we have a condition that we must have destroy a solulu to escape. However, when destroying a solulu tree , we will teleport to a random tree.
         Therefore, i have came up the multiverse idea by creating two graphs and combining them to find the shortest distance from start to end.
         The first graph will contain the treemap specified in the input except we will not have an exit node. The second graph will be the same as the first graph but this will have the exit nodes and no start nodes.
         These 2 graphs  are stored in the same adjacency list where the index of graph 2  = index of graph 1 + N. Additionally,
         These 2 graphs are connected via the solulu trees with an edge from the solulu tree on first graph, to the vertex on the second graph.
         For example, if breaking a solulu tree teleports you to tree number 5, the edges would connect from the solulu tree in graph 1, to tree number 5 in graph 2.
          These edges we add only move from the 1st graph to the 2nd graph, and it will have the edge weight of the time it takes to chop down the solulu tree.
          Therefore, traversing this edge can be seen as chopping the solulu tree. Once we contructed this layered graph structure, we use dijkstra to find the shortest path from the start to the exit.
           The shortest path taken are depended on how many weight (w) is accumulate from roads taken between tree and
        time taken to destroy a solulu tree and teleport .
        Input:
            start: the start means the start note on the treemap
            exits: the exits means the multiple end notes on the treemap
        Return:
            a tuple containing ( the time taken from traveling and destroying solulu tree, the start note on the treemap, the end note that it took to escape)

        Time complexity:
            Best case analysis:O(RLog(T)) ,where T is the number of vertices and R is the number of edges,
                               since we iterate over each list of inputs once.
            Worst case analysis:O(RLog(T)) , same explaination as the best case
        Space complexity:
            Input space analysis:O(|T| + |R|) , the space required to store the T which are the number of verticles and R which are the number of edges .
            Aux space analysis:O(|T| + |R|) , the space required to create the graph with T which are the number of verticles and R which  are the number of edges .
        """
        # Reset all vertices to initial state for fresh computation
        self.reset()
        # Mark designated exits in the graph
        self.set_exits(exits)
        # put the graph into the priority queue
        priority_queue = [self.graph[start]]
        # Starting point distance set to 0
        self.graph[start].shortest_distance = 0
        # Main loop to process vertices according to Dijkstra's algorithm
        while priority_queue:
            # Extract vertex with the smallest distance
            current_vertex = heapq.heappop(priority_queue)
            # If it's an exit, return the shortest distance and send it to the backtrack function to backtrack its path
            if current_vertex.exit:
                return (current_vertex.shortest_distance, self.backtrack_path(current_vertex))
            # Mark vertex as visited
            current_vertex.visited = True
            for edge in current_vertex.edges:
                # if the edges that lead to end vertex is not visited
                if not edge.end.visited:
                    # updating the current shortest distance when there is a  shorter distance
                    if current_vertex.shortest_distance + edge.weight < edge.end.shortest_distance:
                        edge.end.shortest_distance = current_vertex.shortest_distance + edge.weight
                        edge.end.previous_vertex = current_vertex
                        # Add it to the priority_queue and edges used travel from that vertex to that vertex shortest distance has been updated
                        heapq.heappush(priority_queue, edge.end)

    def reset(self):
        """
            Resets all vertices in the graph to their initial unvisited state .


            Input: vertices visted

            Return: Vertices initial set




        Time complexity:
            Best case analysis: 0(N) , where N is the number of vertices.
            Worst case analysis:O(N) , iterating through each vertex.
        Space complexity:
            Input space analysis: O(1), operates directly on existing graph structure.
            Aux space analysis: 0(1), direct modifications without additional storage.
        """
        # the vertices in the graph
        for vertex in self.graph:
            # visit status is reset
            vertex.visited = False
            # set the reset vertex the shortest distance
            vertex.shortest_distance = float('inf')
            # Clear path leading to this vertex
            vertex.previous_vertex = None
            # Reset exit status
            vertex.exit = False


    def set_exits(self, list_exits):
        """
            Marks the specified vertices as exits in the graph.

            Input : give a set of exits

            Return : those are the only exits in the graph



        Time complexity:
            Best case analysis: 0(N) , where N is the number of exits.
            Worst case analysis:O(N) , iterating through the list of exits.
        Space complexity:
            Input space analysis:O(N) , operates on the provided list of exits.
            Aux space analysis: O(1) , modifies existing vertices.
        """
        for end in list_exits:
            # Mark corresponding vertices in the second graph as exits
            self.graph[end + self.N].exit = True


    def backtrack_path(self, end_vertex):
        """
            Constructs the path from the start vertex to the given end vertex by backtracking from the end.

            Input : the path it used to exit

            Return : the reversed way it went from exit to start




        Time complexity:
            Best case analysis:O(N), where N is the path length.
            Worst case analysis: O(N), traversing back through the path.
        Space complexity:
            Input space analysis:O(1), uses the given end vertex.
            Aux space analysis: O(N), to store the path.
        """
        # path is a exit
        path = [end_vertex.id]


        # when the exit vertex is not none
        while end_vertex is not None:
            #check if the second graph  have a exit vertex or not
            if path[-1] != end_vertex.id:
                # Add end vertex to path
                path.append(end_vertex.id)
                # Move to the previous end vertex in the path
            end_vertex = end_vertex.previous_vertex
        # Return reversed path to show the route from start to end
        return path[::-1]











