## @package algorithms.meloalgorithm
# @brief contains the all VNE-NLF algorithms

from gurobipy import *
import logging
import time
import literals
import algorithms.abstract_algorithm
import vnet_tum_wrapper

GUROBI_CODE_DESCRIPTION_SHORT = {
    GRB.LOADED: 'LOADED',
    GRB.OPTIMAL: 'OPTIMAL',
    GRB.INFEASIBLE: 'INFEASIBLE',
    GRB.INF_OR_UNBD: 'INF_OR_UNBD',
    GRB.UNBOUNDED: 'UNBOUNDED',
    GRB.CUTOFF: 'CUTOFF',
    GRB.ITERATION_LIMIT: 'ITERATION_LIMIT',
    GRB.NODE_LIMIT: 'NODE_LIMIT',
    GRB.TIME_LIMIT: 'TIME_LIMIT',
    GRB.SOLUTION_LIMIT: 'SOLUTION_LIMIT',
    GRB.INTERRUPTED: 'INTERRUPTED',
    GRB.NUMERIC: 'NUMERIC',
    GRB.SUBOPTIMAL: 'SUBOPTIMAL',
    GRB.INPROGRESS: 'INPROGRESS'
}

GUROBI_CODE_DESCRIPTION_LONG = {
    GRB.LOADED: 'Model is loaded, but no solution information is available.',
    GRB.OPTIMAL: 'Model was solved to optimality (subject to tolerances), and '
                 'an optimal solution is available.',
    GRB.INFEASIBLE: 'Model was proven to be infeasible.',
    GRB.INF_OR_UNBD: 'Model was proven to be either infeasible or unbounded.'
                     'To obtain a more definitive conclusion, set the'
                     'DualReductions parameter to 0 and reoptimize.',
    GRB.UNBOUNDED: 'Model was proven to be unbounded. Important note: an '
                   'unbounded status indicates the presence of an unbounded'
                   'ray that allows the objective to improve without limit.'
                   'It says nothing about whether the model has a feasible'
                   'solution. If you require information on feasibility,'
                   'you should set the objective to zero and reoptimize.',
    GRB.CUTOFF: 'Optimal objective for model was proven to be worse than the'
                'value specified in the Cutoff parameter. No solution'
                'information is available.',
    GRB.ITERATION_LIMIT: 'Optimization terminated because the total number of'
                         'simplex iterations performed exceeded the value'
                         'specified in the IterationLimit parameter, or'
                         'because the total number of barrier iterations'
                         'exceeded the value specified in the BarIterLimit'
                         'parameter.',
    GRB.NODE_LIMIT: 'Optimization terminated because the total number of'
                    'branch-and-cut nodes explored exceeded the value specified'
                    'in the NodeLimit parameter.',
    GRB.TIME_LIMIT: 'Optimization terminated because the time expended exceeded'
                    'the value specified in the TimeLimit parameter.',
    GRB.SOLUTION_LIMIT: 'Optimization terminated because the number of'
                        'solutions found reached the value specified in the '
                        'SolutionLimit parameter.',
    GRB.INTERRUPTED: 'Optimization was terminated by the user.',
    GRB.NUMERIC: 'Optimization was terminated due to unrecoverable numerical'
                 'difficulties.',
    GRB.SUBOPTIMAL: 'Unable to satisfy optimality tolerances; a sub-optimal'
                    'solution is available.',
    GRB.INPROGRESS: 'An asynchronous optimization call was made, but the'
                    'associated optimization run is not yet complete.'
}

GUROBI_SUCCESS_STATE = [
    GRB.OPTIMAL,
    GRB.ITERATION_LIMIT,
    GRB.NODE_LIMIT,
    GRB.TIME_LIMIT,
    GRB.SOLUTION_LIMIT
]

logger = logging.getLogger('algo-logger')
logger.setLevel(logging.INFO)


## @class MeloAlgorithm
# @brief Base class for the different VNE-NLF algorithms
# @author Michael Manhart
#
# Contains most functions that are used for the VNE-NLF algorithm. Call the
# run() functin of one of the derived classes to run the algorithm.
class MeloAlgorithm(algorithms.abstract_algorithm.AbstractAlgorithm):
    # # @brief constructor
    # @param [in]       alpha: The alpha value that is used by the algorithm
    # @param [in]       beta: The beta value that is used by the algorithm
    # @param[in,out]    physicalNetwork: The physicalnetwork. The cpu and
    #                   bandwidth data will be updated, when the run method is
    #                   called.
    #
    # Sets the virtual member variable to reference the virtual network and sets
    # the values for alpha and beta. Also initializes all other member variables
    def __init__(self, alpha, beta, physicalNetwork, timeout=10, **kwargs):
        super(MeloAlgorithm, self).__init__(physical_network=physicalNetwork)
        ## The virtual network
        self.virtual = None
        ## The Gurobi Model that is used for the optimization
        self.model = Model()
        ## List of all used variables
        self.Vars = dict()

        # Parameters
        self.alpha = alpha
        self.beta = beta
        self.run_time = 0
        if timeout == 'infinity':
            self.timeout = GRB.INFINITY
        else:
            self.timeout = timeout
        self._optimization_time = 0
        self._setup_time = 0

    # # @brief creates all variables that are needed for the embedding process
    #
    # This function creates the x_i_m variables that are used to map nodes
    # and also y_i_j_m_n variables to map edges. For each y_i_j_m_n, that is created
    # also a y_j_i_m_n will be created

    def get_decision_of_binary_variable(self, gurobiVariable):
        self.check_for_binary_type(gurobiVariable)
        return gurobiVariable.x > 0.5

    def check_for_binary_type(self, gurobiVariable):
        if ((gurobiVariable.getAttr("VType") != "B" and
                     gurobiVariable.getAttr("VType") != "I") or
                (gurobiVariable.getAttr("LB") != 0.0 or
                         gurobiVariable.getAttr("UB") != 1.0)):
            raise Exception("Given variable is not binary!",
                            " Name: ",
                            gurobiVariable.getAttr("VarName"),
                            " VType ",
                            gurobiVariable.getAttr("VType"),
                            " LB ",
                            gurobiVariable.getAttr("LB"),
                            " UB ",
                            gurobiVariable.getAttr("UB"))
        return True

    def createVar(self):
        # add x variables for node mapping
        for i in self.physical.getNodes():
            for m in self.supportedVirtualNodes(i):
                self.Vars["x_" + str(i) + "_" + str(m)] = self.model.addVar(vtype=GRB.BINARY,
                                                                            name="x_" + str(i) + "_" + str(m))
        # add y variables for edge mapping
        for (i, j) in self.physical.getEdges():
            for (m, n) in self.supportedVirtualEdges(i, j):
                varStr = "y_" + str(i) + \
                         "_" + str(j) + \
                         "_" + str(m) + \
                         "_" + str(n)
                self.Vars[varStr] = self.model.addVar(vtype=GRB.BINARY, \
                                                      name=varStr)
                self.Vars["y_" + str(i) + "_" + str(j) + "_" + str(m) + \
                          "_" + str(n)] = self.model.addVar(vtype=GRB.BINARY, \
                                                            name="y_" + str(i) + "_" + str(j) + "_" + str(m) + \
                                                                 "_" + str(n))
                self.Vars["y_" + str(j) + "_" + str(i) + "_" + str(m) + "_" + str(n)] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name="y_" + str(j) + "_" + str(i) + "_" + str(m) + "_" + str(n))  # other direction

        # update the model
        self.model.update()

    def supportedVirtualNodes(self, i):
        nodeList = list()

        if self.physical.getCPU(i) < 0:
            # if the physical node has no remaining cpu always return an empty
            # list
            return nodeList

        for m in self.virtual.getNodes():
            if self.virtual.getCPU(m) <= self.physical.getCPU(i):
                nodeList.append(m)

        return nodeList

    def supportedVirtualEdges(self, i, j):
        edgeList = list()

        if self.physical.getBandwidth(i, j) <= 0:
            # if the physical node has no remaining cpu always return an empty
            # list
            return edgeList

        for (m, n) in self.virtual.getEdges():
            if self.virtual.getBandwidth(m, n) < \
                    self.physical.getBandwidth(i, j):
                edgeList.append((m, n))

        return edgeList

    def supportedPhysicalNodes(self, m):
        nodeList = list()

        for i in self.physical.getNodes():
            if (self.virtual.getCPU(m) <= self.physical.getCPU(i)) and \
                    (self.physical.getCPU(i) >= 0):
                nodeList.append(i)

        return nodeList

    def supportedPhysicalEdges(self, m, n):
        edgeList = list()

        for (i, j) in self.physical.getEdges():
            if (self.virtual.getBandwidth(m, n) < \
                        self.physical.getBandwidth(i, j)) and \
                    (self.physical.getBandwidth(i, j) > 0):
                edgeList.append((i, j))

        return edgeList

    def constraintNodeAssignment(self):
        # constraint no. 1
        for m in self.virtual.getNodes():
            expr = LinExpr()
            for i in self.supportedPhysicalNodes(m):
                expr.addTerms(1, self.Vars["x_" + str(i) + "_" + str(m)])
            self.model.addConstr(expr, GRB.EQUAL, 1)

    def constraintOneVirtualNodePerPhysical(self):
        # constraint no. 2
        for i in self.physical.getNodes():
            expr = LinExpr()
            for m in self.supportedVirtualNodes(i):
                expr.addTerms(1, self.Vars["x_" + str(i) + "_" + str(m)])
            self.model.addConstr(expr, GRB.LESS_EQUAL, 1)

    def constraintCPUConversation(self):
        # constraint no. 3
        """for i in self.physical.getNodes():
            expr = LinExpr()
            for m in self.supportedVirtualNodes(i):
                expr.addTerms(self.virtual.getCPU(m), self.Vars["x_" + str(i) + "_" + str(m)])
            self.model.addConstr(expr, GRB.LESS_EQUAL, self.physical.getCPU(i))"""

    def constraintVirtualNodeDistance(self):
        # constraint no. 4
        K = 999999999
        # K is a big constant
        for (m, n) in self.virtual.getEdges():
            for i in self.supportedPhysicalNodes(n):
                lhs = LinExpr()
                for j in self.supportedPhysicalNodes(m):
                    lhs.addTerms(self.physical.getDistance(i, j), self.Vars["x_" + str(j) + "_" + str(m)])
                rhs = LinExpr()
                rhs.addTerms(self.virtual.getDistance(m, n), self.Vars["x_" + str(i) + "_" + str(n)])
                # the last part of this inequation was formated in a way that is supported by the addTerms and addConstant() function of LinExpr
                rhs.addTerms((-1 * K), self.Vars["x_" + str(i) + "_" + str(n)])
                rhs.addConstant(K)
                self.model.addConstr(lhs, GRB.LESS_EQUAL, rhs)

    def constraintMultiCommodity(self):
        # constraint no. 5
        for (m, n) in self.virtual.getEdges():
            for i in self.physical.getNodes():
                lhs = LinExpr()
                for (i2, j) in self.physical.getConnectedEdges(i):
                    # i2 is used here instead of i to make sure, that the value of i is not overridden
                    if (i2, j) in self.supportedPhysicalEdges(m, n) or (j, i2) in self.supportedPhysicalEdges(m, n):
                        lhs.addTerms(1, self.Vars["y_" + str(i2) + "_" + str(j) + "_" + str(m) + "_" + str(n)])
                        lhs.addTerms(-1, self.Vars["y_" + str(j) + "_" + str(i2) + "_" + str(m) + "_" + str(n)])
                rhs = LinExpr()
                if i in self.supportedPhysicalNodes(m):
                    rhs.addTerms(1, self.Vars["x_" + str(i) + "_" + str(m)])
                if i in self.supportedPhysicalNodes(n):
                    rhs.addTerms(-1, self.Vars["x_" + str(i) + "_" + str(n)])
                self.model.addConstr(lhs, GRB.EQUAL, rhs)

    def constraintBandwidth(self):
        # constraint no. 6
        for (i, j) in self.physical.getEdges():
            lhs = LinExpr()
            for (m, n) in self.supportedVirtualEdges(i, j):
                lhs.addTerms(self.virtual.getBandwidth(m, n),
                             self.Vars["y_" + str(i) + "_" + str(j) + "_" + str(m) + "_" + str(n)])
                lhs.addTerms(self.virtual.getBandwidth(m, n),
                             self.Vars["y_" + str(j) + "_" + str(i) + "_" + str(m) + "_" + str(n)])
            self.model.addConstr(lhs, GRB.LESS_EQUAL, self.physical.getBandwidth(i, j))

    def constraintLinkDelay(self):
        # constraint no. 7
        for (m, n) in self.virtual.getEdges():
            for i in self.physical.getNodes():
                lhs = LinExpr()
                for (i, j) in self.physical.getConnectedEdges(i):
                    if (i, j) in self.supportedPhysicalEdges(m, n) or (j, i) in self.supportedPhysicalEdges(m, n):
                        lhs.addTerms(self.physical.getDelay(i, j),
                                     self.Vars["y_" + str(i) + "_" + str(j) + "_" + str(m) + "_" + str(n)])
                        lhs.addTerms(self.physical.getDelay(i, j),
                                     self.Vars["y_" + str(j) + "_" + str(i) + "_" + str(m) + "_" + str(n)])
                self.model.addConstr(lhs, GRB.LESS_EQUAL, self.virtual.getDelay(m, n))

    # # @brief starts the optimization
    # @returns      0
    def optimize(self):
        # time_limit = 10
        # postponed_count = 0
        while True:
            self.model.optimize()
            self.run_time += self.model.getAttr('Runtime')
            if self.model.status == GRB.TIME_LIMIT:
                if self.model.getAttr('MIPGap') == GRB.INFINITY and \
                                self.model.getAttr('Runtime') < 0:
                    continue
                else:
                    break
            else:
                break

        return 0

    # # @brief Writes the result to the graph
    # returns 0 on success, 1 if an error occurred
    def writeResultsToGraph(self):
        # check feasible
        node_embedding = {}
        edge_embedding = {}
        if self.model.status == GRB.INFEASIBLE:
            vnr_classification = literals.VNR_INFEASIBLE
        elif self.model.status == GRB.INF_OR_UNBD:
            self._logger.info('Gurobi says model is infeasible or unbounded')
            vnr_classification = literals.VNR_INFEASIBLE
        elif self.model.status == GRB.INTERRUPTED:
            raise KeyboardInterrupt
        elif self.model.status not in GUROBI_SUCCESS_STATE:
            raise RuntimeError('Unanticipated Result State: {} --> {}'.format(
                self.model.status,
                GUROBI_CODE_DESCRIPTION_SHORT[self.model.status]
            ))
        elif self.model.getAttr('MIPGap') == GRB.INFINITY:
            vnr_classification = literals.VNR_NO_SOLUTION
        else:
            vnr_classification = literals.VNR_ACCEPTED
            for i in self.physical.getNodes():
                for m in self.supportedVirtualNodes(i):
                    var1 = self.get_decision_of_binary_variable(self.Vars["x_" + str(i) + "_" + str(m)])
                    if var1:
                        node_embedding[m] = i

            # edges
            for (i, j) in self.physical.getEdges():
                for (m, n) in self.supportedVirtualEdges(i, j):
                    var1 = self.get_decision_of_binary_variable(
                        self.Vars["y_" + str(i) + "_" + str(j) + "_" + str(m) + "_" + str(n)])
                    var2 = self.get_decision_of_binary_variable(
                        self.Vars["y_" + str(j) + "_" + str(i) + "_" + str(m) + "_" + str(n)])
                    if var1 or var2:
                        if (m,n) not in edge_embedding:
                            edge_embedding[(m, n)] = [(i,j)]
                        else:
                            edge_embedding[(m,n)].append((i,j))

        return self.create_embedding_object(
            vnr_classification=vnr_classification,
            nodes=node_embedding,
            edges=edge_embedding
        )

    def setObjective(self):
        raise NotImplementedError

    # # @brief Executes the embedding
    # @param[in,out] vn: The vn that will be embedded.
    # @returns 0 on sucess
    #
    #
    # This function handels the complete embeding. It will create all variables
    # and constraint and use the setObjective() function of the inherited class
    # to create an objective function. It will then perfom the optimization
    # using the gurobi optimizer and write the results in the vn parameter and
    # update the phyiscal network.
    def run(self, virtual_network, num_cores, **kwargs):
        """ Execute algorithm.

        Args:
            virtual_network (network_model.vne.networkmodel.VirtualNetwork):
                network to embedd.
            num_cores (int): Number of core to run on.
            **kwargs: Additional arguments.

        Returns:
            embedding (scenario.simulation.vne.simulation_objects.Embedding)

        """
        start = time.clock()
        # reset the model
        self.run_time = 0
        self.model = Model()
        self.Vars = dict()
        self._logger = logging.getLogger(str(self.__class__))
        self._logger.setLevel(logging.DEBUG)

        # set the virtual network
        self.virtual = vnet_tum_wrapper.VnettumWrapper(network=virtual_network.to_fnss_topology())

        # create Variables
        self.createVar()

        # set all constraints
        self.constraintNodeAssignment()
        self.constraintOneVirtualNodePerPhysical()
        self.constraintCPUConversation()
        self.constraintVirtualNodeDistance()
        self.constraintMultiCommodity()
        self.constraintBandwidth()
        self.constraintLinkDelay()

        # set the objective function
        self.setObjective()
        stop = time.clock()

        # Set Time stopping criterion and MIP Gap
        self.model.setParam('TimeLimit', self.timeout)
        self.model.setParam('Threads', num_cores)
        self.model.setParam('LogToConsole', 0)

        self._setup_time = stop - start
        start = time.clock()
        # run the optimization
        self.optimize()
        stop = time.clock()
        self._optimization_time = stop - start

        # write to graph
        embedding = self.writeResultsToGraph()
        embedding.optimality_gap = self.model.getAttr('MIPGap')
        embedding.setup_time = self._setup_time
        embedding.solving_time = self._optimization_time
        embedding.run_time = self._setup_time + self._optimization_time
        return embedding


## @class MeloSDP
# @brief Melo's algorithm with SDP objective function
#
# For details see the documentation of the MeloAlgorithm class
class MeloSDP(MeloAlgorithm):
    ## @brief sets the objective function
    def setObjective(self):
        expr = LinExpr()
        for m in self.virtual.getNodes():
            for i in self.supportedPhysicalNodes(m):
                expr.addTerms(float(self.alpha) / (self.physical.getCPU(i)), self.Vars["x_" + str(i) + "_" + str(m)])
        for (m, n) in self.virtual.getEdges():
            for (i, j) in self.supportedPhysicalEdges(m, n):
                expr.addTerms(float(self.beta) / (self.physical.getBandwidth(i, j)),
                              self.Vars["y_" + str(i) + "_" + str(j) + "_" + str(m) + "_" + str(n)])
                expr.addTerms(float(self.beta) / (self.physical.getBandwidth(i, j)),
                              self.Vars["y_" + str(j) + "_" + str(i) + "_" + str(m) + "_" + str(n)])
        self.model.setObjective(expr, GRB.MINIMIZE)


## @class MeloWSDP
# @brief Melo's algorithm with WSDP objective function
#
# For details see the documentation of the MeloAlgorithm class
class MeloWSDP(MeloAlgorithm):
    def setObjective(self):
        expr = LinExpr()
        for m in self.virtual.getNodes():
            for i in self.supportedPhysicalNodes(m):
                expr.addTerms(self.alpha * self.virtual.getCPU(m) / \
                              (self.physical.getCPU(i) + 0.000000000001),
                              self.Vars["x_" + str(i) + "_" + str(m)])
        for (m, n) in self.virtual.getEdges():
            for (i, j) in self.supportedPhysicalEdges(m, n):
                expr.addTerms((float(self.beta) * \
                               self.virtual.getBandwidth(m, n)) / \
                              (self.physical.getBandwidth(i, j) + \
                               0.000000000001), self.Vars["y_" + str(i) + \
                                                          "_" + str(j) + \
                                                          "_" + str(m) + \
                                                          "_" + str(n)])
                expr.addTerms((float(self.beta) * \
                               self.virtual.getBandwidth(m, n)) / \
                              (self.physical.getBandwidth(i, j) + \
                               0.000000000001), self.Vars["y_" + str(j) + \
                                                          "_" + str(i) + \
                                                          "_" + str(m) + \
                                                          "_" + str(n)])
        self.model.setObjective(expr, GRB.MINIMIZE)


## @class MeloLB
# @brief Melo's algorithm with LB objective function
#
# For details see the documentation of the MeloAlgorithm class
class MeloLB(MeloAlgorithm):
    def setObjective(self):
        self.Vars["L_C_max"] = \
            self.model.addVar(vtype=GRB.CONTINUOUS, name="L_C_max")  # maximum Node Load
        self.Vars["L_B_max"] = \
            self.model.addVar(vtype=GRB.CONTINUOUS, name="L_B_max")  # maximum Link Load

        self.model.update()

        # Set value for maximum node load
        for i in self.physical.getNodes():
            expr = LinExpr()
            expr.addConstant(float(self.physical.getCPU(i)) / \
                             self.physical.getMaxCPU(i))
            for m in self.supportedVirtualNodes(i):
                expr.addTerms(float(self.virtual.getCPU(m)) / \
                              self.physical.getMaxCPU(i),
                              self.Vars["x_" + str(i) + "_" + str(m)])
            self.model.addConstr(expr, GRB.LESS_EQUAL, self.Vars["L_C_max"])

        # Set value for maximum link load
        for (i, j) in self.physical.getEdges():
            expr = LinExpr()
            expr.addConstant(float(self.physical.getBandwidth(i, j)) / \
                             self.physical.getMaxBandwidth(i, j))
            for m, n in self.supportedVirtualEdges(i, j):
                expr.addTerms(float(self.virtual.getBandwidth(m, n)) / \
                              self.physical.getMaxBandwidth(i, j),
                              self.Vars["y_" + str(i) + "_" + str(j) + \
                                        "_" + str(m) + "_" + str(n)])
                expr.addTerms(float(self.virtual.getBandwidth(m, n)) / \
                              self.physical.getMaxBandwidth(i, j),
                              self.Vars["y_" + str(j) + "_" + str(i) + \
                                        "_" + str(m) + "_" + str(n)])
            self.model.addConstr(expr, GRB.LESS_EQUAL, self.Vars["L_B_max"])

        # set the objective function
        expr = LinExpr()
        expr.addTerms(self.alpha, self.Vars["L_C_max"])
        expr.addTerms(self.beta, self.Vars["L_B_max"])

        self.model.setObjective(expr, GRB.MINIMIZE)
