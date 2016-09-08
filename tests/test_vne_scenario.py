import logging
import scenario.concrete_scenarios_factories
import input_output.data_interface.bridge.datamodel as datamodel
import scenario.simulation.strategy as vne_strategy
import scenario.simulation.concrete_simulations as sml
import scenario.concrete_scenarios as scs
import unittest
import fnss
import algorithms.modifiedmelo
import literals
import input_output.data_interface.interface_config as interface_config

__author__ = 'Patrick Kalmbach, Michael Manhart'


class AbstractTest(unittest.TestCase):
    @classmethod
    def _setup_database(cls):
        with datamodel.db.atomic() as txn:
            try:
                datamodel.db.create_tables(
                    [
                        datamodel.AlgorithmSetting,
                        datamodel.ProcessSetting,
                        datamodel.NetworkGenerationSetting,
                        datamodel.EventGenerationSetting,
                        datamodel.ProcessToEventSetting,
                        datamodel.Network,
                        datamodel.Node,
                        datamodel.Edge,
                        datamodel.EventHeap,
                        datamodel.Experiment,
                        datamodel.LearningModel,
                        datamodel.NetworkGenerationToNetwork,
                        datamodel.Event,
                        datamodel.Scenario,
                        datamodel.RunConfiguration
                    ],
                    safe=True
                )
                cls.gurobi_setting = {
                    'num_cores': 1
                }
                cls.algo_setting = {
                    'name': literals.ALGORITHM_MELO_SDP,
                    "alpha": 0.5, "beta": 0.5, "timeout": 300
                }
                parameter = {"alpha": 0.5, "beta": 0.5, "timeout": 300}

                cls.ap_setting = {
                    'arrival_rate':5,
                    'distribution':'poisson',
                    'num_requests':10,
                    'type': 'arrival'
                }
                cls.sp_setting = {
                    'arrival_rate':1000,
                    'distribution':'negative_exponential',
                    'type': 'service'
                }
                cls.network_generation_setting =  {
                    'model': literals.NETWORK_MODEL_ERDOES_RENYI,
                    'connectivity': 0.5,
                    'min_order': 5,
                    'max_order': 15,
                    'min_capacity': 0,
                    'max_capacity': 50,
                    'min_cpu': 0,
                    'max_cpu': 50,
                    'capacity_generation': 'uniform',
                    'is_substrate': False,
                    'min_edge_distance': 10,
                    'max_edge_distance': 100,
                    'delay_per_km': 0.05
                }
                cls.substrate_setting = {
                    'model': 'ER',
                    'connectivity': 0.1,
                    'order': 50,
                    'min_capacity': 50,
                    'max_capacity': 100,
                    'min_cpu': 50,
                    'max_cpu': 100,
                    'capacity_generation': 'uniform',
                    'is_substrate': True,
                    'min_edge_distance': 10,
                    'max_edge_distance': 100,
                    'delay_per_km': 0.05
                }
                cls.learning_model = {
                    'type': 'testmode',
                    'serialized_model': '/path/to/model.pkl',
                    'aux_params': 'empty',
                    'comment': 'This model is created from setup method of unittest'
                }
                cls.ap_setting_record = datamodel.ProcessSetting.create(**cls.ap_setting)
                cls.sp_setting_record = datamodel.ProcessSetting.create(**cls.sp_setting)
                cls.network_generation_setting_record = datamodel.NetworkGenerationSetting.create(
                    **cls.network_generation_setting.copy()
                )

                cls.event_generation_setting =  {
                    #'arrival_process_setting': ap_setting.process_setting_id,
                    #'service_process_Setting': sp_setting.process_setting_id,
                    'network_setting': cls.network_generation_setting_record.network_generation_setting_id
                }

                cls.event_generation_setting_record = datamodel.EventGenerationSetting.create(
                    **cls.event_generation_setting
                )
                datamodel.ProcessToEventSetting.create(
                    event_generation_setting=cls.event_generation_setting_record,
                    process_setting=cls.ap_setting_record
                )
                datamodel.ProcessToEventSetting.create(
                    event_generation_setting=cls.event_generation_setting_record,
                    process_setting=cls.sp_setting_record
                )

                cls.substrate_setting_record = datamodel.NetworkGenerationSetting.create(
                    **cls.substrate_setting.copy()
                )

                cls.learning_model_record = datamodel.LearningModel.create(
                    **cls.learning_model
                )
                cls.algo_setting_record = datamodel.AlgorithmSetting.create(
                    name=cls.algo_setting['name'],
                    parameter=str(parameter)
                )
                cls.experiment = datamodel.Experiment.create(
                    description='Created for test purposes from setup method of unittest'
                )
                txn.commit()
            except Exception as e:
                txn.rollback()
                cls.tearDownClass()
                raise e

    @classmethod
    def setUpClass(cls):

        cls.data_source_config = interface_config.PeeweeInterfaceConfig(
            database='TestDb',
            host='127.0.0.1',
            port=3306,
            user='root',
            passwd='root'
        )
        cls.input_if = datamodel.ConnectionManager(cls.data_source_config)
        cls.input_if.connect()

        cls.num_runs = 2
        cls._setup_database()

        cls.logger = logging.getLogger(str(cls.__class__))
        cls.logger.setLevel(logging.DEBUG)

    @classmethod
    def tearDownClass(cls):
        datamodel.db.drop_tables(
            [
                datamodel.AlgorithmSetting,
                datamodel.ProcessSetting,
                datamodel.NetworkGenerationSetting,
                datamodel.EventGenerationSetting,
                datamodel.ProcessToEventSetting,
                datamodel.Network,
                datamodel.Node,
                datamodel.Edge,
                datamodel.EventHeap,
                datamodel.Experiment,
                datamodel.LearningModel,
                datamodel.NetworkGenerationToNetwork,
                datamodel.Event,
                datamodel.Scenario,
                datamodel.RunConfiguration
            ]
        )


class VneScenarioTest(AbstractTest):
    def scenario_creation(self):
        scenarios = scenario.concrete_scenarios_factories.ScenariosFactory.produce(
            problem_type='vne',
            raw_scenarios=self.rawscenarios,
            if_input=self.input_if,
            if_output='Not_needed_for_scenario_creation_test'
        )
        # num arrival processes * num algos
        self.assertEqual(
            len(scenarios),
            len(self.rawscenarios['algorithm_timeout']) * len(self.rawscenarios['arrival_rate']),
            'Wrong number of scenarios created. Expected {} got {}'.format(
                len(self.rawscenarios['algorithm_timeout']) *
                len(self.rawscenarios['arrival_rate']),
                len(scenarios)
            )
        )

        self.rawscenarios.pop('learningmodel_path')
        scenarios = scenario.concrete_scenarios_factories.ScenariosFactory.produce(
            problem_type='vne',
            raw_scenarios=self.rawscenarios,
            if_input=self.input_if,
            if_output='Not_needed_for_scenario_creation_test'
        )
        # num arrival processes * num algos
        self.assertEqual(
            len(scenarios),
            len(self.rawscenarios['algorithm_timeout']) * len(self.rawscenarios['arrival_rate']),
            'Wrong number of scenarios created. Expected {} got {}'.format(
                len(self.rawscenarios['algorithm_timeout']) *
                len(self.rawscenarios['arrival_rate']),
                len(scenarios)
            )
        )

    def test_get_network_generation_settings(self):
        scenario = scs.VneScenario(
            if_input=datamodel.ConnectionManager(self.data_source_config),
            if_output=None,
            num_runs=2,
            substrate_generation_settings=self.substrate_setting,
            vnr_generation_settings=self.network_generation_setting,
            algorithm=self.algo_setting,
            arrival_process=self.ap_setting,
            service_process=self.sp_setting,
            gurobi_settings=None,
            learning_model_settings=self.learning_model
        )
        ngid = scenario._get_network_generation_settings(self.substrate_setting.copy())
        self.assertEqual(
            ngid.identifier,
            self.substrate_setting_record.network_generation_setting_id
        )

        substrate_setting = {
            'model': 'ER',
            'connectivity': 0.3,
            'order': 50,
            'min_capacity': 50,
            'max_capacity': 100,
            'min_cpu': 50,
            'max_cpu': 100,
            'capacity_generation': 'uniform',
            'is_substrate': True,
            'delay_per_km': 0.05
        }
        ngid2 = scenario._get_network_generation_settings(substrate_setting)
        self.assertNotEqual(
            ngid.identifier,
            ngid2.identifier
        )

    def test_get_substrates(self):
        scenario = scs.VneScenario(
            if_input=datamodel.ConnectionManager(self.data_source_config),
            if_output=None,
            num_runs=2,
            substrate_generation_settings=self.substrate_setting,
            vnr_generation_settings=self.network_generation_setting,
            algorithm=self.algo_setting,
            arrival_process=self.ap_setting,
            service_process=self.sp_setting,
            gurobi_settings=None,
            learning_model_settings=self.learning_model
        )
        sids = scenario._get_substrates()
        sids = [s.identifier for s in sids]
        self.assertEqual(len(sids), self.num_runs)

        sids2 = scenario._get_substrates()
        sids2 = [s.identifier for s in sids2]
        self.assertListEqual(sids, sids2)

    def test_get_process(self):
        scenario = scs.VneScenario(
            if_input=datamodel.ConnectionManager(self.data_source_config),
            if_output=None,
            num_runs=2,
            substrate_generation_settings=self.substrate_setting,
            vnr_generation_settings=self.network_generation_setting,
            algorithm=self.algo_setting,
            arrival_process=self.ap_setting,
            service_process=self.sp_setting,
            gurobi_settings=None,
            learning_model_settings=self.learning_model
        )
        process = scenario._get_process(self.ap_setting.copy())
        self.assertEqual(process.identifier, self.ap_setting_record.process_setting_id)
        self.assertEqual(process.arrival_rate, self.ap_setting['arrival_rate'])
        self.assertEqual(process.distribution, self.ap_setting['distribution'])
        self.assertEqual(process.type, self.ap_setting['type'])
        self.assertEqual(process.num_requests, self.ap_setting['num_requests'])

        process = scenario._get_process(self.sp_setting.copy())
        self.assertEqual(process.identifier, self.sp_setting_record.process_setting_id)

        ap_setting = self.ap_setting.copy()
        ap_setting['arrival_rate'] = 10
        sp_setting = self.sp_setting.copy()
        sp_setting['arrival_rate'] = 2000

        process = scenario._get_process(ap_setting.copy())
        self.assertNotEqual(process.identifier, self.ap_setting_record.process_setting_id)

        process = scenario._get_process(sp_setting.copy())
        self.assertNotEqual(process.identifier, self.sp_setting_record.process_setting_id)

        count = datamodel.ProcessSetting.select().count()
        self.assertEqual(count, 4)

    def test_get_algorithm(self):
        scenario = scs.VneScenario(
            if_input=datamodel.ConnectionManager(self.data_source_config),
            if_output=None,
            num_runs=2,
            substrate_generation_settings=self.substrate_setting,
            vnr_generation_settings=self.network_generation_setting,
            algorithm=self.algo_setting,
            arrival_process=self.ap_setting,
            service_process=self.sp_setting,
            gurobi_settings=None,
            learning_model_settings=self.learning_model
        )
        algorithm = scenario._get_algorithm()
        self.assertEqual(algorithm.identifier, self.algo_setting_record.algorithm_setting_id)

        algo_setting = {
            'name': literals.ALGORITHM_MELO_SDP,
            "alpha": 0.5, "beta": 0.5, "timeout": 600
        }

        scenario._algorithm = algo_setting
        algorithm2 = scenario._get_algorithm()
        self.assertNotEqual(algorithm.identifier, algorithm2.identifier)

    def test_get_event_generation(self):
        scenario = scs.VneScenario(
            if_input=datamodel.ConnectionManager(self.data_source_config),
            if_output=None,
            num_runs=2,
            substrate_generation_settings=self.substrate_setting,
            vnr_generation_settings=self.network_generation_setting,
            algorithm=self.algo_setting,
            arrival_process=self.ap_setting,
            service_process=self.sp_setting,
            gurobi_settings=None,
            learning_model_settings=self.learning_model
        )
        arrival_process = scenario._get_process(self.ap_setting)
        service_process = scenario._get_process(self.sp_setting)
        virtual_network_generation = scenario._get_network_generation_settings(
            self.network_generation_setting
        )
        record = scenario._get_event_generation(
            service_process=service_process,
            arrival_process=arrival_process,
            network_generation=virtual_network_generation
        )
        local_ap = self.ap_setting.copy()
        local_ap['arrival_rate'] = 10
        local_sp = self.sp_setting.copy()
        local_sp['arrival_rate'] = 2000
        netgen = self.network_generation_setting.copy()
        netgen['min_order'] = 20
        netgen['max_order'] = 30

        arrival_process2 = scenario._get_process(local_ap)
        service_process2 = scenario._get_process(local_sp)
        virtual_network_generation2 = scenario._get_network_generation_settings(netgen)

        ids = [record.identifier]

        record2 = scenario._get_event_generation(
            service_process=service_process2,
            arrival_process=arrival_process,
            network_generation=virtual_network_generation
        )
        self.assertNotIn(record2.identifier, ids)
        ids.append(record2.identifier)
        record3 = scenario._get_event_generation(
            service_process=service_process,
            arrival_process=arrival_process2,
            network_generation=virtual_network_generation
        )
        self.assertNotIn(record3.identifier, ids)
        ids.append(record3.identifier)
        record4 = scenario._get_event_generation(
            service_process=service_process,
            arrival_process=arrival_process,
            network_generation=virtual_network_generation2
        )
        self.assertNotIn(record4.identifier, ids)
        ids.append(record2.identifier)

        count = datamodel.EventGenerationSetting.select().count()
        self.assertEqual(count, 4)

    def test_get_event_queues(self):
        scenario = scs.VneScenario(
            if_input=datamodel.ConnectionManager(self.data_source_config),
            if_output=None,
            num_runs=2,
            substrate_generation_settings=self.substrate_setting,
            vnr_generation_settings=self.network_generation_setting,
            algorithm=self.algo_setting,
            arrival_process=self.ap_setting,
            service_process=self.sp_setting,
            gurobi_settings=None,
            learning_model_settings=self.learning_model
        )
        arrival_process = scenario._get_process(self.ap_setting)
        service_process = scenario._get_process(self.sp_setting)
        virtual_network_generation = scenario._get_network_generation_settings(
            self.network_generation_setting
        )
        event_generation = scenario._get_event_generation(
            service_process=service_process,
            arrival_process=arrival_process,
            network_generation=virtual_network_generation
        )

        queues = scenario._get_event_queues(
            service_process=service_process,
            arrival_process=arrival_process,
            network_generation=virtual_network_generation,
            event_generation=event_generation
        )
        ids1 = [q.identifier for q in queues]
        self.assertEqual(len(ids1), self.num_runs)

        queues2 = scenario._get_event_queues(
            service_process=service_process,
            arrival_process=arrival_process,
            network_generation=virtual_network_generation,
            event_generation=event_generation
        )
        ids2 = [q.identifier for q in queues2]
        self.assertEqual(len(ids2), self.num_runs)
        self.assertListEqual(ids1, ids2)

        num_events = datamodel.Event.select().count()
        num_vnrs = datamodel.Network.select() \
            .where(datamodel.Network.subclass_type == literals.CLASS_VIRTUAL_NETWORK)\
            .count()

        self.assertEqual(num_events, self.num_runs * self.ap_setting['num_requests'])
        self.assertEqual(num_events, num_vnrs)

        orders = range(
            self.network_generation_setting['min_order'],
            self.network_generation_setting['max_order']
        )
        for vnr in datamodel.Network.select().where(datamodel.Network.subclass_type == literals.CLASS_VIRTUAL_NETWORK).execute():
            self.assertIn(vnr.num_nodes, orders)
            self.assertEqual(vnr.model, literals.NETWORK_MODEL_ERDOES_RENYI,
                             'model was not ER but {}'.format(vnr.model))

    def test_simulation_configuration(self):
        scenario1 = scs.VneScenario(
            if_input=datamodel.ConnectionManager(self.data_source_config),
            if_output=None,
            num_runs=self.num_runs,
            substrate_generation_settings=self.substrate_setting.copy(),
            vnr_generation_settings=self.network_generation_setting.copy(),
            algorithm=self.algo_setting.copy(),
            arrival_process=self.ap_setting.copy(),
            service_process=self.sp_setting.copy(),
            gurobi_settings=self.gurobi_setting.copy(),
            learning_model_settings=self.learning_model.copy()
        )
        scenario2 = scs.VneScenario(
            if_input=datamodel.ConnectionManager(self.data_source_config),
            if_output=None,
            num_runs=self.num_runs,
            substrate_generation_settings=self.substrate_setting,
            vnr_generation_settings=self.network_generation_setting,
            algorithm=self.algo_setting,
            arrival_process=self.ap_setting,
            service_process=self.sp_setting,
            gurobi_settings=self.gurobi_setting,
            learning_model_settings=self.learning_model
        )
        scenario1.configure_simulations(literals.WRITE_AT_END_STRATEGY)
        scenario2.configure_simulations(literals.WRITE_AT_END_STRATEGY)

        num_runs = datamodel.RunConfiguration.select().count()
        num_substrates = datamodel.Network.select().where(
            datamodel.Network.subclass_type == literals.CLASS_PHYSICAL_NETWORK
        ).count()
        num_queues = datamodel.EventHeap.select().count()
        self.assertEqual(num_runs, self.num_runs)
        self.assertEqual(num_substrates, self.num_runs)
        self.assertEqual(num_queues, self.num_runs)
        self.assertEqual(
            len(scenario1.simulation_config),
            len(scenario2.simulation_config)
        )

        for i in range(len(scenario1.simulation_config)):
            self.assertDictEqual(
                scenario1.simulation_config[i],
                scenario2.simulation_config[i]
            )


class VneStrategyTest(AbstractTest):
    def setUp(self):
        datamodel.db.create_tables(
            [
                datamodel.Run,
                datamodel.EventOccurrence,
                datamodel.Embedding
            ]
        )
        self._setup_existing = self.setups[0]
        self._setup_missing = self.setups[1]
        self._run = datamodel.Run.create(setup=self._setup_existing.setup_id)

    def tearDown(self):
        datamodel.db.drop_tables(
            [
                datamodel.Embedding,
                datamodel.EventOccurrence,
                datamodel.Run
            ]
        )

    def test_strategy_init_existing_setup(self):
        strategy = vne_strategy.StrategyFactory.produce('write_at_end')
        simulation = sml.VneSimulation(
            sim_strategy=strategy,
            data_source=self.input_if,
            data_sinks=[self.input_if],
            substrate_id=self._setup_existing.substrate.network_id,
            algorithm_id=self._setup_existing.algorithm_setting.algorithm_setting_id,
            event_heap_id=self._setup_existing.event_heap.event_heap_id,
            learning_model_id=self._setup_existing.learning_model_id
        )
        strategy.simulation = simulation
        strategy.initialize()
        self.assertEqual(
            strategy._setup.setup_id,
            self._setup_existing.setup_id,
            (
                'IDs for setups do not match. Expected {} got {}'.format(
                    self._setup_existing.setup_id,
                    strategy._setup.setup_id,
                )
            )
        )
        self.assertIsInstance(
            strategy._algorithm,
            algorithms.vne.modifiedmelo.MeloSDP,
            (
                'Algorithm has wrong class. Expected {} but got {}'.format(
                    str(algorithms.vne.modifiedmelo.MeloSDP),
                    str(strategy._algorithm.__class__)
                )
            )
        )
        self.assertIsInstance(
            strategy._substrate,
            fnss.Topology,
            (
                'Substrate of strategy is wrong type. Expected {} found {}'.format(
                    str(fnss.Topology),
                    str(strategy._substrate.__class__)
                )
            )
        )

    def test_strategy_init_missing_setup(self):
        strategy = vne_strategy.StrategyFactory.produce('write_at_end')
        network_id = self._setup_existing.substrate.network_id
        algo_id = self._setup_existing.algorithm_setting.algorithm_setting_id
        heap_id = self._setup_existing.event_heap.event_heap_id
        model_id = self._setup_existing.learning_model_id
        self._setup_missing.delete_instance()
        simulation = sml.VneSimulation(
            sim_strategy=strategy,
            data_source=self.input_if,
            data_sinks=[self.input_if],
            substrate_id=network_id,
            algorithm_id=algo_id,
            event_heap_id=heap_id,
            learning_model_id=model_id
        )
        strategy.simulation = simulation
        strategy.initialize()
        self.assertEqual(
            strategy._setup.algorithm_setting_id,
            algo_id,
            (
                'IDs for algos do not match. Expected {} got {}'.format(
                    algo_id,
                    strategy._setup.algorithm_setting_id,
                )
            )
        )
        self.assertEqual(
            strategy._setup.substrate_id,
            network_id,
            (
                'IDs for substrates do not match. Expected {} got {}'.format(
                    self._setup_existing.substrate_id,
                    network_id,
                )
            )
        )
        self.assertEqual(
            strategy._setup.event_heap_id,
            heap_id,
            (
                'IDs for Event Heaps do not match. Expected {} got {}'.format(
                    heap_id,
                    strategy._setup.event_heap_id
                )
            )
        )
        self.assertEqual(
            strategy._setup.learning_model_id,
            int(model_id),
            (
                'IDs for learning models do not match. Expected {} got {}'.format(
                    int(model_id),
                    int(strategy._setup.learning_model_id)
                )
            )
        )
        self.assertIsInstance(
            strategy._algorithm,
            algorithms.vne.modifiedmelo.MeloSDP,
            (
                'Algorithm has wrong class. Expected {} but got {}'.format(
                    str(algorithms.vne.modifiedmelo.MeloSDP),
                    str(strategy._algorithm.__class__)
                )
            )
        )
        self.assertIsInstance(
            strategy._substrate,
            fnss.Topology,
            (
                'Substrate of strategy is wrong type. Expected {} found {}'.format(
                    str(fnss.Topology),
                    str(strategy._substrate.__class__)
                )
            )
        )


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(VneScenarioTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
