from numpy import ndarray


class Individual:
    """Class that defines an individual in the GA population"""

    # =========================================
    # ============== CONSTRUCTOR ==============
    # =========================================
    def __init__(self, initialAlphaValues: ndarray):
        """Constructor of an Individual instance"""
        # Alpha Values (a.k.a. the genotype of the individual)
        self.alphaValues = initialAlphaValues
        # True Cost (a.k.a. the fitness of the individual)
        self.isSolved = False  # Flip true when a relaxed-LP solver runs/returns solution data; flip false when the alpha values array is modified
        self.trueCost = 0.0
        self.fakeCost = 0.0
        # Expressed Network (a.k.a. the phenotype of the individual)
        self.arcFlows = {}
        self.arcsOpened = {}
        self.srcFlows = []
        self.sinkFlows = []
        self.paths = []  # Data structure for topology-based operators

    # =========================================
    # ============== METHODS ==============
    # =========================================
    def setAlphaValues(self, alphaValues: ndarray) -> None:
        """Resets the alpha values to a new array"""
        self.alphaValues = alphaValues

    def resetCostValues(self) -> None:
        """Resets just the cost values (i.e. fitness) of the individual"""
        self.isSolved = False
        self.trueCost = 0.0
        self.fakeCost = 0.0

    def resetOutputNetwork(self) -> None:
        """Resets the expressed network (i.e. phenotype) output data in the individual"""
        self.isSolved = False
        self.arcFlows = {}
        self.arcsOpened = {}
        self.srcFlows = []
        self.sinkFlows = []
        self.paths = []
        self.trueCost = 0.0
        self.fakeCost = 0.0
