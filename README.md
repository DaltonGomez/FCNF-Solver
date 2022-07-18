# FCNF-Solver

FCNF-Solver is a sandbox codebase for the NP-Hard *Fixed Charge Network Flow Problem* (FCNF).

## Problem Description
FCNF is a network design problem that optimizes the topology and routing of a commodity over an input graph, returning 
the flow network solution as output. The input comprises a target demand (i.e. total amount of flow the output network 
must realize) and an undirected graph of nodes and edges. Each node is identified as either a source, sink, or 
intermediate node, making FCNF a specific type of the transshipment problem. Each edge in the graph has a capacity 
(i.e. max flow it can be assigned), a variable cost (i.e. a cost paid per unit of flow assigned to the edge), and a 
fixed cost (i.e. an all-or-nothing cost paid if any amount of flow is assigned to the edge). FCNF is an optimization 
problem that minimizes the total cost of the network, subject to meeting the target demand, abiding by the capacity 
constraints of the edges, and maintaining conservation of flow from source to sink. FCNF formulated as a mixed-integer 
linear program (MILP) is given below:

**Compute:**

$$ \min \sum_{e_{ij} \in E} v_{ij} q_{ij} + f_{ij} y_{ij} $$


**Subject to:**

1) $$ y_{ij} \in \lbrace 0,1 \rbrace, \quad \forall e_{ij} \in E $$

2) $$ \sum_{i \in T} t_i \geq d, \quad \forall i \in T $$

3) $$ 0 \leq q_{ij} \leq c_{ij} y_{ij}, \quad \forall e_{ij} \in E $$

4) $$ \sum_{j:e_{ij} \in E} q_{ij}-\sum_{j:e_{ji} \in E} q_{ji} =
      \begin{cases}
          s_i, \text{if}\ j \in S\\
          -t_i, \text{if}\ j \in T\\
          0, \text{otherwise}\\
      \end{cases} 
    , \quad \forall j \in N $$

In this formulation, $N$ represents the set of all nodes, $S$ is the set of all sources, and $T$ is the set of all 
sinks. $s_i$ and $t_i$ is the assigned flow produced or consumed at source or sink $i$, respectively. $E$ is the set of 
all edges, where edge $e_{ij}$ spans from node $i$ to node $j$. (Note that, since the input graph is undirected, the 
edge $e_{ij}$ implies an edge  $e_{ji}$.) For each edge $e_{ij}$ in $E$, $v_{ij}$ is the variable cost of the edge, 
$f_{ij}$ is the fixed cost of the edge, and $c_{ij}$ is the capacity of the edge. These are input parameters of the 
graph to be solved. The decision variables of FCNF are $q_{ij}$, the amount of flow in $\mathbb{R}^+$ assigned to edge 
$e_{ij}$, and $y_{ij}$, the binary decision to open edge $e_{ij}$ and pay the full fixed cost. Constraint #1 forces 
$y_{ij}$ to be binary and, therefore, discretizes the search space, causing this formulation to be a MILP. $d$ 
is the target demand of the network set by the user, which is enforced by Constraint #2. Constraint #3 ensures that an
edge is opened before it can be used and that no edge exceeds its capacity. Constraint #4 maintains conservation of
flow by allowing only sources to produce flow, sinks to consume flow, and intermediate nodes to transport flow. Lastly,
the objective function minimizes the total cost of the returned flow network.

The returned output of the above optimization problem is the set of directed edges used and the amount of flow assigned 
to each used edge. In this sense, FCNF allows us to model "from-scratch" transportation infrastructure as the act of 
opening an edge (i.e. paying the full fixed cost) represents the construction of the infrastructure, and the act of 
assigning flow to each edge (i.e. paying the variable cost proportional to the use) represents the operation and 
maintenance of the infrastructure. FCNF, therefore, provides a useful abstraction that disregards the physical commodity
being transported. Applications of FCNF vary, including cars on roads, water in pipes, and data in broadband networks.
The specific use case motivating this repository is carbon capture and storage infrastructure, which is widely proposed 
but not yet constructed. Additionally, several generalizations of FCNF exist. Source and sink nodes can take on their 
own capacities and/or costs, resulting in Source/Sink-Capacitated FCNF and Source/Sink-Charged FCNF, respectively. By
allowing a discrete number of parallel arcs per edge, where each arc has its own capacity and costs, the problem becomes
the Parallel-Arc FCNF. 

## Repo Overview

### 

**TODO - Update Structure and Class Descriptions**

+ src
  + Graph
    + CandidateGraph.py
    + Node.py
    + Arc.py
    + GraphGenerator.py
    + GraphVisualizer.py
  + FlowNetwork
    + FlowNetworkSolution.py
    + SolutionVisualizer.py
  + Solvers
    + MILPsolverCPLEX.py
    + AlphaSolverPDLP.py
  + AlphaGenetic
    + Population.py
    + Individual.py
    + AlphaSolver.py
  + Experiments
    + GAvsGA.py
    + GAvsMILP.py
    + HyperparamTuner.py
    + MultiGAvsMILP.py
+ data
  + graph_instances
  + SimCCS_models
  + solution_instances

## Project Dependencies

**TODO - Add/Describe Dependencies**

Python 3.8, CPLEX, Numpy, MatPlotLib, SciPy, etc.

## Tests

**TODO - Implement Tests and Add Descriptions**
