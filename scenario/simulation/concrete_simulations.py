import logging
import input_output.data_interface.interface_factory as interface_factory
import input_output.data_interface.interface_config as interface_config
import simulation_objects as simulation_objects
import network_model.networkmodel as vne_networkmodel
from input_output.data_interface.bridge.implementor_factory import ImplementorFactory


class BaseSimulation(object):
    def __init__(self, sim_strategy=None, data_source=None, data_sinks=None):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.strategy = sim_strategy
        self.data_source = data_source
        self.setup = None
        self.metrics_calculator = None
        self.data_sinks = data_sinks
        self.results = []
        self.stop = False

        # The simulation factory will set these factories
        self.algorithm_factory = None
        self.strategy_factory = None
        self.substrate_factory = None

    def run(self):
        raise NotImplementedError

    def set_algorithm_factory(self, concrete_algorithm_factory):
        self.algorithm_factory = concrete_algorithm_factory

    def set_strategy_factory(self, concrete_strategy_factory):
        self.strategy_factory = concrete_strategy_factory

    def set_substrate_factory(self, concrete_substrate_factory):
        self.substrate_factory = concrete_substrate_factory


class VneSimulation(BaseSimulation):
    def __init__(self, sim_strategy, substrate_id, algorithm_id, event_heap_id,
                 learning_model_id, scenario_id, run_configuration_id,
                 data_source=None, data_sinks=None, data_source_config=None,
                 data_sink_configs=None, num_cores=1):
        """ Initializes object.

            Args:
                sim_strategy (scenario.simulation.vne.strategy.BaseStrategy): Simulation strategy.
                data_source (input_output.data_interface.vne.db_interface, optional):
                    Interface to retrieve data from.
                data_sinks (input_output.data_interface.vne.db_interface, optional): list,
                    Interface to write simulation results to.
                substrate_id (int): Database ID of substrate to use for simulation.
                algorithm_id (int): Database ID of algorithm to use for simulation.
                event_heap_id (int) Database ID of event heap to use for simulation.
                learning_model_id (int): Databae ID of learning model to use
                    for simulation.
                scenario_id (int): Database ID of scenario.
                run_configuration_id (int): Database ID of run configuration
                data_source_config (dict): Configuration for input data source.
                data_sink_configs (list): Configuration for data sink
                    configurations. List of dicts.

            Raises:
                AssertionError, if both, data_source/sinks, data_source_config/
                    data_sink_configs is not set.

            Note:
                `data_source`, `,data_sink`, may be used, if simulation is
                executed on the local machine and it is possible to simply
                pass objects along.
                `data_source_config` and `data_sink_configs` should be used in
                conjunction with celery. In the Docs it is discouraged of passing
                actual objects/class references to a worker. Thus only the
                settings are passed over and the interfaces will be created in
                the `initialize` step of the scenario.
                If both are set, existing interfaces is given precedence.
        """
        super(VneSimulation, self).__init__(
            sim_strategy,
            data_source,
            data_sinks
        )
        assert \
            (data_source is not None) or (data_source_config is not None), \
            'Either data_source or data_source_config must be set. Both were None'
        assert \
            (data_sinks is not None) or (data_sink_configs is not None), \
            'Either data_sinks or data_sink_configs must be set. Both were None'
        self.substrate_id = substrate_id
        self.event_heap_id = event_heap_id
        self.algorithm_id = algorithm_id
        self.learning_model_id = learning_model_id
        self.scenario_id = scenario_id
        self.run_configuration_id = run_configuration_id
        self.num_cores = num_cores
        self.data_source_config = data_source_config
        self.data_sink_configs = data_sink_configs
        self.embeddings = []
        self.substrate_states = []
        self.occurred_events = []
        self.current_state = None
        # self.current_vnr = None
        self.current_embedding = None
        self.event_queue = None
        self.substrate = None
        self.simulation_steps = 0

    def _initialize(self):
        """ Retrieve objects for simulation from datasource.

            Note:
                Running a simulation is split into two parts, configuration
                and execution. During configuration, parameter from the config
                file are translated into simulation objects wich are then
                persisted to a data sink (if_interface).
                During simulation, the objects are recovered from where they
                have been written to.
                Reason for this two phase step is, that the configuration happens
                centrally and the simulations themselves are executed on different
                machines, so configuration must be made accessible to those
                machines.
        """
        # These two if clauses are necessary when using celery. Celery does not
        # work well with objects/class references, therefore store the connection
        # parameter and re-initialize the connections on the clients
        if (self.data_source_config is not None) and \
                (self.data_source is None):
            config = interface_config.InterfaceConfigFactory.produce(
                parameter=self.data_source_config
            )
            self.data_source = interface_factory.InterfaceFactory.produce(
                problem_type='vne',
                interface_config=config
            )
        if (self.data_sink_configs is not None) and \
                (self.data_sinks is None):
            self.data_sinks = []
            for sink in self.data_sink_configs:
                config = interface_config.InterfaceConfigFactory.produce(
                    parameter=sink
                )
                self.data_sinks.append(
                    interface_factory.InterfaceFactory.produce(
                        problem_type='vne',
                        interface_config=config
                    )
                )

        self.substrate = vne_networkmodel.PhysicalNetwork.from_datasource(
            implementor=ImplementorFactory.produce(
                interface=self.data_source,
                object_class=vne_networkmodel.PhysicalNetwork
            ),
            object_id=self.substrate_id
        )
        algorithm = simulation_objects.Algorithm.from_database(
            implementor=ImplementorFactory.produce(
                interface=self.data_source,
                object_class=simulation_objects.Algorithm
            ),
            object_id=self.algorithm_id
        )
        self.algorithm = algorithm
        #if self.learning_model_id in [-1, None]:
        #    self.algorithm = algorithm
        #else:
        #    learning_model = simulation_objects.BrezeSupervisedRnnModel.from_database(
        #        implementor=ImplementorFactory.produce(
        #            interface=self.data_source,
        #            object_class=simulation_objects.BrezeSupervisedRnnModel
        #        ),
        #        object_id=self.learning_model_id
        #    )
        #    self.algorithm = vne_algorithm_factory.AlgorithmFactory.produce(
        #        name='RNN_FILTER',
        #        substrate=None,
        #        learning_model=learning_model,
        #        successor=algorithm
        #    )
        self.algorithm.physical = self.substrate

        self.event_queue = simulation_objects.EventQueue.from_database(
            implementor=ImplementorFactory.produce(
                interface=self.data_source,
                object_class=simulation_objects.EventQueue
            ),
            object_id=self.event_heap_id
        )
        self.run_execution = simulation_objects.RunExecution(
            implementor=ImplementorFactory.produce(
                interface=self.data_source,
                object_class=simulation_objects.RunExecution
            )
        )
        self.run_execution.stage_of_execution = 1
        self.run_execution.num_cores = self.num_cores
        for sink in self.data_sinks:
            implementor = ImplementorFactory.produce(
                interface=sink,
                object_class=simulation_objects.RunExecution
            )
            self.run_execution.implementor = implementor
            self.run_execution.save(run_configuration_id=self.run_configuration_id)

        self.strategy.run_execution = self.run_execution

    def run(self):
        self._initialize()
        # 1. loop over event heap
        # 2. Handle Event
        #   2.1. Arrival Event
        #       2.1.1. Filter request
        #       2.1.2. On success, embedd it
        #       2.1.3. Apply embedding to substrate
        #       2.1.4. Append Embedding to list
        #       2.1.5. Update run execution
        #   2.2 Departure Event
        #       2.2.1. Remove embedding from substrate
        # 3. Create substrate state and append it to list
        for event in self.event_queue:
            self.simulation_steps += 1
            occurrence = event.handle(self)
            occurrence.implementor = ImplementorFactory.produce(
                interface=self.data_source,
                object_class=simulation_objects.EventOccurrence
            )

            if self.current_embedding is not None:
                self.current_embedding.implementor = ImplementorFactory.produce(
                    interface=self.data_source,
                    object_class=simulation_objects.Embedding
                )

            self.current_state = simulation_objects.SubstrateState(
                substrate=self.substrate,
                implementor=ImplementorFactory.produce(
                    interface=self.data_source,
                    object_class=simulation_objects.SubstrateState
                )
            )

            result = simulation_objects.SimulationStepResult(
                embedding=self.current_embedding,
                substrate_state=self.current_state,
                event_occurrence=occurrence
            )
            self.strategy.append_simulation_step(result)
        self.run_execution.stage_of_execution = 2
        self.strategy.write()

