
from src.Experiments.MILP import MILP

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_milp.py

LINUX RUN COMMAND:
cd Repos/FCNF-Solver/
python3 run_milp.py
"""

if __name__ == "__main__":
    # Graph
    inputGraph = "small_9"
    # Solver
    milp = MILP(inputGraph, isTimeConstrained=False, timeLimit=-1.0, isOneArcPerEdge=True,
                isDrawing=True, isLabeling=True, isGraphing=True, isOutputtingCPLEX=True, isSolutionSaved=True)
    milp.solveGraph()
