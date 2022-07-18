# FCNF-Solver

## Problem Description
FCNF-Solver is a sandbox codebase for the NP-Hard *Fixed Charge Network Flow Problem* (FCNF). FCNF is a 
network design problem that optimizes the topology and routing of a commodity over an input graph, returning the flow 
network solution as output. The input comprises a target demand (i.e. total amount of flow the output network must 
realize) and an undirected graph of nodes and edges. Each node is identified as either a source, sink, or intermediate 
node, making FCNF a specific type of the transshipment problem. Each edge in the graph has a capacity (i.e. max flow it 
can be assigned), a variable cost (i.e. a cost paid per unit of flow assigned to the edge), and a fixed cost (i.e. an 
all-or-nothing cost paid if any amount of flow is assigned to the edge). FCNF is an optimization problem that minimizes
the total cost of the network, subject to meeting the target demand, abiding by the capacity constraints of the edges,
and maintaining conservation of flow from source to sink. FCNF formulated as a mixed-integer linear program (MILP) is 
given below:

**Compute:**

$$ \min \sum_{(ij)\in E} v_{ij} q_{ij} + f_{ij} y_{ij} $$


**Subject to:**

$$ y_{ij} \in \{ 0,1 \}, \quad \forall (ij) \in E $$

$$ \sum_{i \in T} t_i \geq d, \quad \forall i \in T $$

$$ 0 \leq q_{ij} \leq c_{ij} y_{ij}, \quad \forall (ij) \in E $$

$$ \sum_{j:(ij) \in E} q_{ij}-\sum_{j:(ji) \in E} q_{ji} =
      \begin{cases}
          s_i, \text{if}\ j \in S\\
          -t_i, \text{if}\ j \in T\\
          0, \text{otherwise}\\
      \end{cases} 
    , \quad \forall j\in N $$

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
