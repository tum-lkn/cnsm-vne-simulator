'''
Created on Jan 18, 2015

@author: andi
'''

import util
import logging

from gurobipy import *

log = logging.getLogger("gurobiutils")

"""

        NAMING CONVENTIONS

"""


def constraint_name_counter(constraintCounter, name):
    return name + str(constraintCounter)


def transform_variable_name(name):
    if isinstance(name, str):
        return name.replace(" ", "-")
    return name


"""

        ACCESSING SOLUTION INFORMATION

"""


def get_decision_of_binary_variable(gurobiVariable):
    check_for_binary_type(gurobiVariable)
    return gurobiVariable.x > 0.5


def check_for_binary_type(gurobiVariable):
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


"""

        HANDLING PARAMETER (RE-)SETTING

"""

Param_MIPGap = "MIPGap"
Param_IterationLimit = "IterationLimit"
Param_NodeLimit = "NodeLimit"
Param_Heuristics = "Heuristics"
Param_Threads = "Threads"
Param_TimeLimit = "TimeLimit"
Param_MIPFocus = "MIPFocus"
Param_RootCutPasses = "CutPasses"
Param_Cuts = "Cuts"
Param_NodefileStart = "NodefileStart"
Param_NodeMethod = "NodeMethod"
Param_Method = "Method"

_listOfUserVariableParameters = [Param_MIPGap,
                                 Param_IterationLimit,
                                 Param_NodeLimit,
                                 Param_Heuristics,
                                 Param_Threads,
                                 Param_TimeLimit,
                                 Param_Cuts,
                                 Param_MIPFocus,
                                 Param_RootCutPasses,
                                 Param_NodefileStart,
                                 Param_Method,
                                 Param_NodeMethod]


class GurobiSettings(object):
    def __init__(self,
                 mipGap=None,
                 iterationLimit=None,
                 nodeLimit=None,
                 heuristics=None,
                 threads=None,
                 timelimit=None,
                 MIPFocus=None,
                 rootCutPasses=None,
                 cuts=None,
                 BarConvTol=None,
                 OptimalityTol=None,
                 Presolve=None,
                 NodefileStart=None,
                 Method=None,
                 NodeMethod=None):
        util.checkPosFloat(mipGap)
        self.MIPGap = mipGap

        util.checkPosFloat(iterationLimit)
        self.IterationLimit = iterationLimit

        util.checkPosInt(nodeLimit)
        self.NodeLimit = nodeLimit

        util.checkPercentage(heuristics)
        self.Heuristics = heuristics

        util.checkPosInt(threads)
        self.Threads = threads

        util.checkPosFloat(timelimit)
        self.TimeLimit = timelimit

        util.checkIntWithRange(MIPFocus, -1, 3)
        self.MIPFocus = MIPFocus

        self.rootCutPasses = rootCutPasses
        self.cuts = cuts

        self.BarConvTol = BarConvTol
        self.OptimalityTol = OptimalityTol
        self.Presolve = Presolve

        util.checkPosFloat(NodefileStart)
        self.NodefileStart = NodefileStart

        self.Method = Method
        self.NodeMethod = NodeMethod

    def setTimeLimit(self, newTimeLimit):
        util.checkPosFloat(newTimeLimit)
        self.TimeLimit = newTimeLimit


def get_default_gurobisettings():
    gurobiSettings = GurobiSettings(mipGap=None,
                                    iterationLimit=GRB.INFINITY,
                                    nodeLimit=None,
                                    heuristics=None,
                                    threads=1,
                                    timelimit=3600,
                                    MIPFocus=1,
                                    cuts=-1,
                                    rootCutPasses=-1,
                                    Method=-1,
                                    NodeMethod=-1
                                    )
    return gurobiSettings


def apply_gurobi_settings(model, gurobiSettings):
    if gurobiSettings.MIPGap is not None:
        set_param(model, Param_MIPGap, gurobiSettings.MIPGap)
    else:
        reset_parameter(model, Param_MIPGap)

    if gurobiSettings.IterationLimit is not None:
        set_param(model, Param_IterationLimit, gurobiSettings.IterationLimit)
    else:
        reset_parameter(model, Param_IterationLimit)

    if gurobiSettings.NodeLimit is not None:
        set_param(model, Param_NodeLimit, gurobiSettings.NodeLimit)
    else:
        reset_parameter(model, Param_NodeLimit)

    if gurobiSettings.Heuristics is not None:
        set_param(model, Param_Heuristics, gurobiSettings.Heuristics)
    else:
        reset_parameter(model, Param_Heuristics)

    if gurobiSettings.Threads is not None:
        set_param(model, Param_Threads, gurobiSettings.Threads)
    else:
        reset_parameter(model, Param_Heuristics)

    if gurobiSettings.TimeLimit is not None:
        set_param(model, Param_TimeLimit, gurobiSettings.TimeLimit)
    else:
        reset_parameter(model, Param_TimeLimit)

    if gurobiSettings.MIPFocus is not None:
        set_param(model, Param_MIPFocus, gurobiSettings.MIPFocus)
    else:
        reset_parameter(model, Param_MIPFocus)

    if gurobiSettings.cuts is not None:
        set_param(model, Param_Cuts, gurobiSettings.cuts)
    else:
        reset_parameter(model, Param_Cuts)

    if gurobiSettings.rootCutPasses is not None:
        set_param(model, Param_RootCutPasses, gurobiSettings.rootCutPasses)
    else:
        reset_parameter(model, Param_RootCutPasses)

    if gurobiSettings.NodefileStart is not None:
        set_param(model, Param_NodefileStart, gurobiSettings.NodefileStart)
    else:
        reset_parameter(model, Param_NodefileStart)

    if gurobiSettings.Method is not None:
        set_param(model, Param_Method, gurobiSettings.Method)
    else:
        reset_parameter(model, Param_Method)

    if gurobiSettings.NodeMethod is not None:
        set_param(model, Param_NodeMethod, gurobiSettings.NodeMethod)
    else:
        reset_parameter(model, Param_NodeMethod)


def reset_all_parameter(model):
    for param in _listOfUserVariableParameters:
        (name, type, curr, min, max, default) = model.getParamInfo(param)
        model.setParam(param, default)


def reset_parameter(model, param):
    (name, type, curr, min, max, default) = model.getParamInfo(param)
    log.info("ModelCreator: param {0} is set to {1}".format(param, default))
    model.setParam(param, default)


def set_param(model, param, value):
    log.info("ModelCreator: param {0} is set to {1}".format(param, value))
    if not param in _listOfUserVariableParameters:
        raise Exception("You cannot access the parameter <" + param + ">!")
    else:
        model.setParam(param, value)


def get_param(model, param):
    if not param in _listOfUserVariableParameters:
        raise Exception("You cannot access the parameter <" + param + ">!")
    else:
        model.getParam(param)
