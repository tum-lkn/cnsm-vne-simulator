import errors.homebrewederrors as errorbrewer
import concrete_scenarios
import itertools
import literals

__author__ = "Andreas Blenk, Michael Manhart"


class ScenariosFactory(object):
    """
    Factory to create Scenarios suited to the different problem types
    """

    @classmethod
    def produce(cls,
                problem_type,
                raw_scenarios,
                if_input,
                if_output):
        """
        Creates a list of scenarios based on the given configuration. A scenario is created for each element of the
        cartesian product of the lists in raw_scenarios.
        Args:
            problem_type: Type of the problem (dhpp, hpp, cpp, vne)
            raw_scenarios: Parameters of the scenarios as a dict containing lists with the values of the respective
                parameters
            if_input: interface to fetch the input data
            if_output: list of interfaces to store the output

        Returns:
            List with instances of Scenario
            None, if problem_type is '?' or the lists in raw_scenarios are empty
        """

        scenarios = []

        if problem_type == 'vne':
            def cross_product(prefix):
                """ Creates cross product of parameter values for a simulation
                    element, e.g. the arrival process.

                    Args:
                        prefix (string): Specifying the prefix parameter combination
                            should be retrieved.

                    Returns:
                        product (list): list of dictionary, cross product of parameter values.

                    Note:
                        This method allows a dynamic creation of parameter lists.
                        This part of the code is oblivious to changes in config files
                        and such and just passes settings on.
                """
                product = []
                param_names = [] # Contains later all parameter names for prefix
                param_values = []   # Contains all the parameter list for respective
                                    # parameter name
                for param in raw_scenarios:
                    if param.startswith(prefix):
                        param_name = param[len(prefix):]
                        param_names.append(param_name)
                        try:
                            values = [float(p) for p in raw_scenarios[param]]
                        except ValueError as e:
                            values = raw_scenarios[param]
                        finally:
                            param_values.append(values)

                # create the cross product between all specified parameter values
                # the resulting tuple will have the same order as param_values
                # i.e. param_names. Thus by zipping them together we get all
                # configurations we want.
                if len(param_values) > 0:
                    # Only if we actually found sth, else we return a list with
                    # an empty dict in it
                    for values in itertools.product(*param_values):
                        product.append(dict(zip(param_names, values)))
                return product

            substrate_prefix = 'substrate_'
            algo_prefix = 'algorithm_'  # prefix for all algorithm related parameters
            vnr_prefix = 'vnr_'
            gurobi_prefix = 'gurobi_'
            learning_model_prefix = 'learningmodel_'
            arrival_process_prefix = 'ap_'
            service_process_prefix = 'sp_'

            substrate_generation_settings = cross_product(substrate_prefix)
            for process in substrate_generation_settings:
                process['is_substrate'] = True
            algorithms = cross_product(algo_prefix)
            vnr_generation_processes = cross_product(vnr_prefix)
            for process in vnr_generation_processes:
                process['is_substrate'] = False
            gurobi_settings = cross_product(gurobi_prefix)
            learning_model_settings = cross_product(learning_model_prefix)
            arrival_processes = cross_product(arrival_process_prefix)
            for settings in arrival_processes:
                settings['type'] = literals.PROCESS_TYPE_ARRIVAL
            service_processes = cross_product(service_process_prefix)
            for settings in service_processes:
                settings['type'] = literals.PROCESS_TYPE_SERVICE
            num_runs = int(raw_scenarios['num_runs']) if 'num_runs' in raw_scenarios else 1

            if len(learning_model_settings) == 0:
                # In case no settings for learning model were specified
                learning_model_settings = [None]

            assert \
                len(substrate_generation_settings) > 0,\
                'No substrate generation settings found. Check syntax in config' \
                'file. Correct prefix is {}'.format(substrate_prefix)
            assert \
                len(algorithms) > 0, \
                'No algorithm settings found. Check syntax in config file.' \
                'Correct prefix is {}'.format(algo_prefix)
            assert \
                len(vnr_generation_processes) > 0, \
                'No vnr generation process settings found. Check syntax in' \
                'config file. Correct prefix is {}'.format(vnr_prefix)
            assert \
                len(gurobi_settings) > 0, \
                'No gurobi settings found. Check syntax in config file.' \
                'Correct prefix is {}'.format(gurobi_prefix)
            assert \
                len(learning_model_settings) > 0, \
                'No learning model settings found. Check syntax in config file.' \
                'Correct prefix is {}'.format(learning_model_prefix)
            assert \
                len(arrival_process_prefix) > 0, \
                'No arrival process settings found. Check syntax in config file.' \
                'Correct prefix is {}'.format(arrival_process_prefix)
            assert \
                len(service_process_prefix) > 0, \
                'No service process settings found. Check syntax in config file.' \
                'Correct prefix is {}'.format(service_process_prefix)

            iterator = itertools.product(
                substrate_generation_settings,
                algorithms,
                arrival_processes,
                service_processes,
                vnr_generation_processes,
                learning_model_settings,
                gurobi_settings
            )
            for sgs, algo, ap, sp, vnrgs, ls, gs in iterator:
                scenarios.append(concrete_scenarios.VneScenario(
                    if_input,
                    if_output,
                    num_runs=num_runs,
                    substrate_generation_settings=sgs,
                    algorithm=algo,
                    arrival_process=ap,
                    service_process=sp,
                    vnr_generation_settings=vnrgs,
                    learning_model_settings=ls,
                    gurobi_settings=gs
                ))
            return scenarios
        elif problem_type == '?':
            return None
        else:
            raise errorbrewer.ProblemNotKnownError('problem type {} is unknown'.format(problem_type))

