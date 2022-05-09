from src.AntExperiments.AntResultsExperiment import AntResultsExperiment

"""
RUN COMMAND:
cd PycharmProjects/FCNF-Solver
py -3.8 run_ant_result.py
"""

if __name__ == "__main__":
    numAnts = 50
    numEpisodes = 15
    networkList = [
        "25-AntDemo-AntDemo",
    ]

    experiment = AntResultsExperiment(networkList, numAnts, numEpisodes)
    experiment.runExperiment()

"""
# ORIGINAL NETWORK LIST:
networkList = [
    "25-AntDemo-AntDemo",
    "25-AntDemo-5",
    "25-AntDemo-10",
    "50-AntDemo-AntDemo",
    "50-AntDemo-5",
    "50-AntDemo-10",
    "100-AntDemo-AntDemo",
    "100-AntDemo-5",
    "100-AntDemo-10",
    "200-AntDemo-AntDemo",
    "200-AntDemo-5",
    "200-AntDemo-10",
    "300-AntDemo-AntDemo",
    "300-AntDemo-5",
    "300-AntDemo-10",
    "400-AntDemo-AntDemo",
    "400-AntDemo-5",
    "400-AntDemo-10",
    ]
"""

"""
# SECOND NETWORK LIST:
networkList = [
    "25-AntDemo-AntDemo-2",
    "25-AntDemo-5-2",
    "25-AntDemo-10-2",
    "50-AntDemo-AntDemo-2",
    "50-AntDemo-5-2",
    "50-AntDemo-10-2",
    "100-AntDemo-AntDemo-2",
    "100-AntDemo-5-2",
    "100-AntDemo-10-2",
    "200-AntDemo-AntDemo-2",
    "200-AntDemo-5-2",
    "200-AntDemo-10-2",
    "300-AntDemo-AntDemo-2",
    "300-AntDemo-5-2",
    "300-AntDemo-10-2",
    "400-AntDemo-AntDemo-2",
    "400-AntDemo-5-2",
    "400-AntDemo-10-2" 
    ]
"""

"""
# THIRD NETWORK LIST:
networkList = [
    "25-AntDemo-AntDemo-3",
    "25-AntDemo-5-3",
    "25-AntDemo-10-3",
    "50-AntDemo-AntDemo-3",
    "50-AntDemo-5-3",
    "50-AntDemo-10-3",
    "100-AntDemo-AntDemo-3",
    "100-AntDemo-5-3",
    "100-AntDemo-10-3",
    "200-AntDemo-AntDemo-3",
    "200-AntDemo-5-3",
    "200-AntDemo-10-3",
    "300-AntDemo-AntDemo-3",
    "300-AntDemo-5-3",
    "300-AntDemo-10-3",
    "400-AntDemo-AntDemo-3",
    "400-AntDemo-5-3",
    "400-AntDemo-10-3"   
    ]
"""

"""
# NOT RUN
networkList = [
    "500-AntDemo-AntDemo",
    "500-AntDemo-5",
    "500-AntDemo-10",
    "1000-AntDemo-AntDemo",
    "1000-AntDemo-5",
    "1000-AntDemo-10"
    ]
"""
