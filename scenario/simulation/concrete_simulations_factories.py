import scenario.simulation.concrete_simulations
import errors.homebrewederrors as errorbrewer
import input_output.data_interface.interface_config as if_config
from scenario.simulation.strategy import StrategyFactory as VneStrategyFactory


class SimulationFactory(object):
    @classmethod
    def produce(cls, sim_config, if_input, if_output):
        # create simulation object
        if sim_config['problem_type'] == 'vne':
            if 'data_source_config' not in sim_config:
                sim_config['data_source_config'] = None
            if 'data_sink_configs' not in sim_config:
                sim_config['data_sink_configs'] = None
            strategy = VneStrategyFactory.produce(sim_config['sim_strategy'])
            simulation = scenario.simulation.concrete_simulations.VneSimulation(
                sim_strategy=strategy,
                data_source=if_input,
                data_sinks=if_output,
                substrate_id=sim_config['substrate_id'],
                algorithm_id=sim_config['algorithm_id'],
                event_heap_id=sim_config['event_heap_id'],
                learning_model_id=sim_config['learning_model_id'],
                num_cores=sim_config['num_cores'],
                data_source_config=sim_config['data_source_config'],
                data_sink_configs=sim_config['data_sink_configs'],
                run_configuration_id=sim_config['run_configuration_id'],
                scenario_id=sim_config['scenario_id']
            )
            strategy.simulation = simulation
            return simulation
        else:
            raise errorbrewer.ProblemNotKnownError("Problem {}".format(sim_config['problem_type']))

