""" Simple Class wrapping the numeric result value of gurobi into a more easy way to handle.

"""

from gurobipy import *


class GurobiStatus(object):
    LOADED = 1  # Model is loaded, but no solution information is available.
    OPTIMAL = 2  # Model was solved to optimality (subject to tolerances), and an optimal solution is available.
    INFEASIBLE = 3  # Model was proven to be infeasible.
    INF_OR_UNBD = 4  # Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.
    UNBOUNDED = 5  # Model was proven to be unbounded. Important note: an unbounded status indicates the presence of an unbounded ray that allows the objective to improve without limit. It says nothing about whether the model has a feasible solution. If you require information on feasibility, you should set the objective to zero and reoptimize.
    CUTOFF = 6  # Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter. No solution information is available.
    ITERATION_LIMIT = 7  # Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the IterationLimit parameter, or because the total number of barrier iterations exceeded the value specified in the BarIterLimit parameter.
    NODE_LIMIT = 8  # Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the NodeLimit parameter.
    TIME_LIMIT = 9  # Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.
    SOLUTION_LIMIT = 10  # Optimization terminated because the number of solutions found reached the value specified in the SolutionLimit parameter.
    INTERRUPTED = 11  # Optimization was terminated by the user.
    NUMERIC = 12  # Optimization was terminated due to unrecoverable numerical difficulties.
    SUBOPTIMAL = 13  # Unable to satisfy optimality tolerances; a sub-optimal solution is available.
    IN_PROGRESS = 14  # A non-blocking optimization call was made (by setting the NonBlocking parameter to 1 in a Gurobi Compute Server environment), but the associated optimization run is not yet complete.

    def __init__(self, solCount=0, status=1, gap=GRB.INFINITY):
        self.solCount = solCount
        self.status = status
        self.gap = gap

    def isFeasible(self):
        result = self.solCount > 0
        if self.status == self.INFEASIBLE:
            result = False
        if self.status == self.INF_OR_UNBD:
            result = False
        if self.status == self.UNBOUNDED:
            result = False
        if self.status == self.NUMERIC:
            result = False
        if self.gap == GRB.INFINITY:
            result = False
        return result

    def isOptimal(self):
        if self.status == self.OPTIMAL:
            return True
        else:
            return False
