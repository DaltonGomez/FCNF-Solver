import numpy as np

from src.FlowNetwork.CandidateGraph import CandidateGraph
from src.FlowNetwork.SolutionVisualizer import SolutionVisualizer
from src.Solvers.MILPsolverCPLEX import MILPsolverCPLEX
from src.Solvers.RelaxedLPSolverPDLP import RelaxedLPSolverPDLP

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_opt_alphas.py
"""

if __name__ == "__main__":
    # Get inputs
    minTargetFlow = 1000
    networkName = "1000-1-10.p"
    network = CandidateGraph()
    network = network.loadCandidateGraph(networkName)

    # Find exact solution
    milp = MILPsolverCPLEX(network, minTargetFlow, isOneArcPerEdge=False)
    exact = milp.findSolution()
    # exact.saveSolution()
    exactVisual = SolutionVisualizer(exact)
    exactVisual.drawLabeledSolution()

    # Reverse engineer "optimal" alpha values
    alphaValues = np.full((network.numEdges, network.numArcsPerEdge), 1.0)
    # For each arc in the exact solution
    for e in range(network.numEdges):
        for c in range(network.numArcsPerEdge):
            flow = exact.arcFlows[(e, c)]
            # If there is a flow, set alpha = 1/flow
            if flow > 0:
                alphaValues[e][c] = 1 / flow
            # Else if there's no flow, set alpha = infinity
            elif flow == 0:
                alphaValues[e][c] = 999999999

    # Solve relaxed LP with "optimal" alpha values
    relaxedLp = RelaxedLPSolverPDLP(network, minTargetFlow)
    relaxedLp.updateObjectiveFunction(alphaValues)
    relaxedLp.solveModel()
    approx = relaxedLp.writeSolution()
    # approx.saveSolution()
    approxVisual = SolutionVisualizer(approx)
    approxVisual.drawLabeledSolution()
