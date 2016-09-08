import simulation.concrete_simulations_factories as factories
import control.celery_control as celery
import re
import logging
import literals
import errors.homebrewederrors as custom_exceptions
import simulation.simulation_objects as vne_sim_objects
import utils.EventQueueGenerator as evtgen
import utils.NetworkGenerator as netgen
import network_model.networkmodel as vne_networkmodel
from input_output.data_interface.bridge.implementor_factory import ImplementorFactory

__authors__ = 'Patrick Kalmbach, Andreas Blenk, Johannes Zerwas'


class Scenario(object):
    """
    A scenario holds the problem specific parameters that characterize a simulation/experiment
    Attributes:
        simulation_config: Additional information to execute the simulations as dict (simulation strategy, celery usage)
        input: interface to fetch the input data
        output: list of interfaces to store the output
        simulations: list that contains the different runs of this scenario
    """

    def __init__(self,
                 if_input,
                 if_output):
        self.input = if_input
        self.output = if_output
        self.simulations = []
        self.simulation_config = []

    # FIXME How can we tell anyone that these are the standard interfaces?
    def configure_simulations(self, sim_strategy):
        raise NotImplementedError

    # FIXME How can we tell anyone that this is the standard celery way?
    def run_celery(self):
        """
        Creates and starts a single simulation using Celery

        Returns:
            None
        """
        for sim_config in self.simulation_config:
            celery.do_simulation.delay(sim_config, self.input, self.output)

    def run(self):
        if len(self.simulations) == 0:
            self.create_simulations()

        for sim in self.simulations:
            sim.run()

    def create_simulations(self):
        for sim_config in self.simulation_config:
            sim = factories.SimulationFactory.produce(
                sim_config,
                if_input=self.input,
                if_output=self.output
            )
            self.simulations.append(sim)


class VneScenario(Scenario):

    def __init__(self, if_input, if_output, num_runs, substrate_generation_settings,
                 vnr_generation_settings, algorithm, arrival_process, service_process,
                 gurobi_settings, learning_model_settings=None,
                 setup_ids=None):
        """ Initializes object.

            Args:
                if_input (): TODO: How does this look like?
                if_output (): TODO: How does this look like?
                learning_model (string, optional): Path to model used for filtering.
                setup_ids (list, optional): List of integers explicitely stating
                    setups that should be executed. If this parameter is set
                    all other are ignored.
        """
        super(VneScenario, self).__init__(if_input, if_output)
        self.num_runs = num_runs
        self._if_input = if_input
        self._if_output = if_output

        self._arrival_process = arrival_process
        self._service_process = service_process
        self._substrate_generation_settings = substrate_generation_settings
        self._algorithm = algorithm
        self._vnr_generation_settings = vnr_generation_settings

        self._learning_model_settings = learning_model_settings
        self._setup_ids = setup_ids
        self._gurobi_settings = gurobi_settings
        self.logger = logging.getLogger(str(self.__class__))
        self.logger.setLevel(logging.DEBUG)

    def _get_substrates(self):
        """ Create a substrate network object and write it to database.
            Intended to be used in conjunction with method `configure_simulation`
            if substrate could not be found or more need to be created.

            Returns:
                network_ids (list): identifier of newly created substrate.
        """
        def create_substrate(generation_settings):
            settings = generation_settings.copy()
            settings['substrate'] = True
            model = settings.pop('model')
            generation_function = netgen.NetworkGenerator.method_factory(model)
            topology = generation_function(**settings)
            substrate = vne_networkmodel.PhysicalNetwork(
                fnss_topology=topology,
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_networkmodel.PhysicalNetwork
                )
            )
            substrate.save(substrate_generation=netgenobj)
            return substrate

        netgenobj = self._get_network_generation_settings(self._substrate_generation_settings)

        try:
            substrate_ids = vne_networkmodel.PhysicalNetwork.get_identifier(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_networkmodel.PhysicalNetwork
                ),
                network_generation=netgenobj
            )
            substrates = []
            for sid in substrate_ids:
                substrates.append(
                    vne_networkmodel.PhysicalNetwork.from_datasource(
                        implementor=ImplementorFactory.produce(
                            interface=self._if_input,
                            object_class=vne_networkmodel.PhysicalNetwork
                        ),
                        object_id=sid
                    )
                )
        except custom_exceptions.SubstrateNotKnownError as e:
            self.logger.info('Could not find a substrate with specified params.'
                             'I will create one for you...')
            substrates = [create_substrate(netgenobj.todict())]

        if self.num_runs is not None:
            if len(substrates) < self.num_runs:
                self.logger.info(
                    (
                        'You requested {} runs, but only {} substrate'
                        'networks have been created. I will create the'
                        'missing ones for you...'
                    ).format(self.num_runs, len(substrates))
                )
                difference = self.num_runs - len(substrates)
                aux_subs = [create_substrate(netgenobj.todict()) for i in range(difference)]
                substrates.extend(aux_subs)
            elif len(substrates) > self.num_runs:
                self.logger.info(
                    (
                        'Found more substrate networks than requested runs.'
                        ' I will exclude {} substrates'
                    ).format(len(substrates) - self.num_runs)
                )
                substrates = substrates[:self.num_runs]
        return substrates

    def _get_process(self, settings):
        try:
            process_id = vne_sim_objects.StochasticProcess.get_identifier(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.StochasticProcess
                ),
                **settings
            )
            process = vne_sim_objects.StochasticProcess.from_database(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.StochasticProcess
                ),
                object_id=process_id
            )
        except custom_exceptions.StochasticProcessNotKnownError:
            self.logger.info((
                                 'No Process with parameter {} found, I will create '
                                 'one for you').format(str(settings))
                             )
            process = vne_sim_objects.StochasticProcess.from_type(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.StochasticProcess
                ),
                **settings
            )
            process.save()
        return process

    def _get_network_generation_settings(self, settings):
        try:
            net_gen_id = vne_sim_objects.NetworkGenerationSettings.get_identifier(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.NetworkGenerationSettings
                ),
                **settings
            )
            net_gen = vne_sim_objects.NetworkGenerationSettings.from_database(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.NetworkGenerationSettings
                ),
                object_id=net_gen_id
            )
        except custom_exceptions.NetworkGenerationSettingsNotKnownError:
            self.logger.info((
                                 'No Network Generation Settings with parameter {} found, I will create '
                                 'one for you').format(str(self._vnr_generation_settings))
                             )
            net_gen = vne_sim_objects.NetworkGenerationSettings(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.NetworkGenerationSettings
                ),
                **settings
            )
            net_gen.save()
        return net_gen

    def _get_event_generation(self, service_process, arrival_process, network_generation):
        try:
            event_generation_id = vne_sim_objects.EventGenerationSettings.get_identifier(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.EventGenerationSettings
                ),
                service_process=service_process,
                arrival_process=arrival_process,
                network_generation_process=network_generation
            )
            event_generation = vne_sim_objects.EventGenerationSettings.from_database(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.EventGenerationSettings
                ),
                object_id=event_generation_id
            )
        except custom_exceptions.EventGenerationSettingsNotKnownError:
            self.logger.debug(
                ('Could not find event generation record, I will create one for you...')
            )
            event_generation = vne_sim_objects.EventGenerationSettings(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.EventGenerationSettings
                ),
                arrival_process=arrival_process,
                service_process=service_process,
                network_generation_settings=network_generation
            )
            event_generation.save()
        return event_generation

    def _get_event_queues(self, service_process, arrival_process,
                             network_generation, event_generation):

        def create_event_heap(arrival_process, service_process, network_generation,
                              event_generation):
            event_generator = evtgen.EventGenerator(
                lmbd=arrival_process.arrival_rate,
                avg_life_time=service_process.arrival_rate,
                model=network_generation.model,
                num_requests=arrival_process.num_requests
            )
            vnrs = event_generator.generate_networks(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_networkmodel.VirtualNetwork
                ),
                **network_generation.todict()
            )
            times = event_generator.generate_event_times()
            events = event_generator.generate_events(vnrs, times)
            event_queue = vne_sim_objects.EventQueue(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.EventQueue
                )
            )
            for event in events:
                event.implementor = ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.ArrivalEvent
                )
                event_queue.heappush(event)
            event_queue.save(event_generation_object=event_generation)
            return event_queue

        try:
            event_heap_ids = vne_sim_objects.EventQueue.get_identifier(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.EventQueue
                ),
                event_generation_id=event_generation.identifier
            )
            event_heaps = []
            for oid in event_heap_ids:
                implementor = ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.EventQueue
                )
                event_heaps.append(vne_sim_objects.EventQueue.from_database(
                    implementor=implementor,
                    object_id=oid
                ))
        except custom_exceptions.EventHeapNotKnownError as e:
            self.logger.info('Could not find an event heap with specified params.'
                             'I will create one for you...')
            event_heap = create_event_heap(
                event_generation=event_generation,
                arrival_process=arrival_process,
                service_process=service_process,
                network_generation=network_generation
            )
            event_heaps = [event_heap]

        if self.num_runs is not None:
            if len(event_heaps) < self.num_runs:
                self.logger.info(
                    (
                        'You requested {} runs, but only {} event heaps'
                        ' have been created. I will create the '
                        'missing ones for you...'
                    ).format(self.num_runs, len(event_heaps))
                )
                difference = self.num_runs - len(event_heaps)
                for i in range(difference):
                    event_heap = create_event_heap(
                        arrival_process=arrival_process,
                        service_process=service_process,
                        network_generation=network_generation,
                        event_generation=event_generation
                    )
                    event_heaps.append(event_heap)
            elif len(event_heaps) > self.num_runs:
                self.logger.info(
                    (
                        'Found more event heaps than requested runs.'
                        ' I will exclude {} event heaps'
                    ).format(len(event_heaps) - self.num_runs)
                )
                event_heaps = event_heaps[:self.num_runs]

        return event_heaps

    def _get_algorithm(self):
        """ Retrieve algorithm id for database or create a new record if no one
            id found.

            Returns:
                 algorithm_id (simulation object): Algorithm object
        """
        local = self._algorithm.copy()
        name = local.pop('name')
        implementor = ImplementorFactory.produce(
            interface=self._if_input,
            object_class=vne_sim_objects.Algorithm
        )
        try:
            algo_id = vne_sim_objects.Algorithm.get_identifier(
                implementor=implementor,
                name=name,
                parameter=local
            )
            algorithm_object = vne_sim_objects.Algorithm.from_database(
                implementor=implementor,
                object_id=algo_id
            )
            algorithm_object.implementor=implementor
        except custom_exceptions.AlgorithmNotKnownError as e:
            self.logger.info('Could not find an algorithm with specified params.'
                             'I will create one for you...')
            algorithm_object = vne_sim_objects.Algorithm(
                name=name,
                parameter=local
            )
            algorithm_object.implementor = implementor
            # Write it to input interface rather than output interface as this
            # part belongs to the setup part and not to the simulation results
            algorithm_object.save()
        return algorithm_object

    def configure_simulations(self, sim_strategy):
        """ Creates Simulations (i,e, runs) for this scenario. The components
            (substrate, algorithm etc) are assumed to already be present in the
            database. Each Run has a setup, collecting the explicite components.
            If this setup is not present in the databse it will be added.
            Also the new run will be added to the database.

            Args:
                sim_strategy (string): Specifies how results should be written
                    to the databse. Must be in `{write_at_end, write_after_#+}`
                    where `#+` is a placeholder specifying after how many handled
                    events a intermediate state should be written to the database.
        """
        regex = '({}|{})'.format(
            literals.WRITE_AT_END_STRATEGY,
            literals.WRITE_AFTER_STRATEGY
        )
        assert re.match(regex, sim_strategy) is not None, \
            'Unknown sim_strategy {}'.format(sim_strategy)

        arrival_process = self._get_process(self._arrival_process)
        service_process = self._get_process(self._service_process)
        virtual_network_generation = self._get_network_generation_settings(
            self._vnr_generation_settings
        )
        event_generation = self._get_event_generation(
            service_process=service_process,
            arrival_process=arrival_process,
            network_generation=virtual_network_generation
        )
        physical_network_generation = self._get_network_generation_settings(
            self._substrate_generation_settings
        )
        #if self._learning_model_settings is None:
        #    learning_model_id = None
        #else:
        #    learning_model_id = vne_sim_objects.BrezeSupervisedRnnModel.get_identifier(
        #        ImplementorFactory.produce(
        #            self._if_input,
        #            vne_sim_objects.BrezeSupervisedRnnModel
        #        ),
        #        **self._learning_model_settings
        #    )
        learning_model_id = None
        algorithm_setting = self._get_algorithm()

        try:
            scenario_id = vne_sim_objects.Scenario.get_identifier(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.Scenario
                ),
                event_generation=event_generation,
                algorithm_setting=algorithm_setting,
                network_generation=physical_network_generation,
                learning_model_id=learning_model_id
            )
            scenario = vne_sim_objects.Scenario.from_database(
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.Scenario
                ),
                object_id=scenario_id
            )
        except custom_exceptions.ScenarioNotKnownError:
            self.logger.info('Scenario does not yet exist, I will create it for you...')
            scenario = vne_sim_objects.Scenario(
                algorithm_setting=algorithm_setting,
                learning_model_id=learning_model_id,
                event_generation=event_generation,
                network_generation=physical_network_generation,
                implementor=ImplementorFactory.produce(
                    interface=self._if_input,
                    object_class=vne_sim_objects.Scenario
                )
            )
            scenario.save()

        substrates = self._get_substrates()
        event_heaps = self._get_event_queues(
            arrival_process=arrival_process,
            service_process=service_process,
            network_generation=virtual_network_generation,
            event_generation=event_generation
        )

        if self.num_runs is not None:
            if len(substrates) > len(event_heaps):
                self.logger.info(
                    (
                        'I found more substrates than event heaps, I will '
                        'ommit {} substrates'.format(
                            len(substrates) - len(event_heaps)
                        )
                    )
                )
                substrates = substrates[:len(event_heaps)]
            else:
                if len(substrates) < len(event_heaps):
                    self.logger.info(
                        (
                            'I found more event heaps than substrates, I will '
                            'ommit the event heaps with ids {}'.format(
                                len(event_heaps) - len(substrates)
                            )
                        )
                    )
                    event_heaps = event_heaps[:len(substrates)]

        run_implementor = ImplementorFactory.produce(
            interface=self._if_input,
            object_class=vne_sim_objects.RunConfiguration
        )
        for sub, evtq in zip(substrates, event_heaps):
            exists_configuration = vne_sim_objects.RunConfiguration.exists(
                scenario=scenario,
                network=sub,
                event_queue=evtq,
                implementor=run_implementor
            )
            if exists_configuration:
                self.logger.info('Run configuration already created, I will skip this one.')
                run_configid = vne_sim_objects.RunConfiguration.get_identifier(
                    scenario=scenario,
                    network=sub,
                    event_queue=evtq,
                    implementor=run_implementor
                )
                obj = vne_sim_objects.RunConfiguration.from_database(
                    implementor=run_implementor,
                    object_id=run_configid
                )
            else:
                obj = vne_sim_objects.RunConfiguration(
                    scenario=scenario,
                    network=sub,
                    event_queue=evtq,
                    implementor=run_implementor
                )
                obj.save()
            self.simulation_config.append({
                'substrate_id': sub.identifier,
                'algorithm_id': algorithm_setting.identifier,
                'event_heap_id': evtq.identifier,
                'learning_model_id': learning_model_id,
                'sim_strategy': sim_strategy,
                'problem_type': 'vne',
                'num_cores': int(self._gurobi_settings['num_cores']),
                'run_configuration_id': obj.identifier,
                'scenario_id': scenario.identifier,
                'data_source_config': self._if_input.as_dict()
            })

    def run_celery(self):
        for sim_config in self.simulation_config:
            sim_config['data_source_config'] = self._if_input.config
            sim_config['data_sink_configs'] = [cfg.config for cfg in self._if_output]
            celery.do_simulation.delay(
                sim_config,
                None,
                None
            )
