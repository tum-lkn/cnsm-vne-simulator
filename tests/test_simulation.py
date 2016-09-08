import scenario.simulation.concrete_simulations as concrete_simulations
import scenario.concrete_scenarios as concrete_scenarios
import scenario.simulation.concrete_simulations_factories as sim_factories
import input_output.data_interface.interface_config as interface_config
import input_output.data_interface.bridge.datamodel as datamodel
import literals
import unittest
import logging


class SimulationTest(unittest.TestCase):

    data_source_config = interface_config.PeeweeInterfaceConfig(
        database='TestDb',
        host='127.0.0.1',
        port=3306,
        user='root',
        passwd='root'
    )
    data_sink_config = interface_config.PeeweeInterfaceConfig(
        database='TestDb',
        host='127.0.0.1',
        port=3306,
        user='root',
        passwd='root'
    )
    num_runs = 1


    gurobi_setting = {
        'num_cores': 1
    }
    algo_setting = {
        'name': literals.ALGORITHM_MELO_SDP,
        "alpha": 0.5, "beta": 0.5, "timeout": 10
    }

    ap_setting = {
        'arrival_rate':5,
        'distribution':'poisson',
        'num_requests': 2,
        'type': 'arrival'
    }
    sp_setting = {
        'arrival_rate':1000,
        'distribution':'negative_exponential',
        'type': 'service'
    }
    network_generation_setting =  {
        'model': literals.NETWORK_MODEL_ERDOES_RENYI,
        'connectivity': 0.5,
        'min_order': 2,
        'max_order': 5,
        'min_capacity': 0,
        'max_capacity': 50,
        'min_cpu': 0,
        'max_cpu': 50,
        'capacity_generation': 'uniform',
        'is_substrate': False,
        'min_edge_distance': 10,
        'max_edge_distance': 100,
        'minx': 100,
        'maxx': 200,
        'miny': 100,
        'maxy': 200,
        'delay_per_km': 0.05
    }
    substrate_setting = {
        'model': 'ER',
        'connectivity': 0.1,
        'order': 20,
        'min_capacity': 50,
        'max_capacity': 100,
        'min_cpu': 50,
        'max_cpu': 100,
        'capacity_generation': 'uniform',
        'is_substrate': True,
        'min_edge_distance': 10,
        'max_edge_distance': 100,
        'minx': 0,
        'maxx': 100,
        'miny': 0,
        'maxy': 100,
        'delay_per_km': 0.05
    }
    learning_model = {
        'type': 'testmode',
        'serialized_model': '/path/to/model.pkl',
        'aux_params': 'empty',
        'comment': 'This model is created from setup method of unittest'
    }

    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger(str(cls.__class__))
        cls.logger.setLevel(logging.DEBUG)
        input_if = datamodel.ConnectionManager(cls.data_source_config)
        input_if.connect()

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
                        datamodel.RunConfiguration,
                        datamodel.RunExecution,
                        datamodel.EventOccurrence,
                        datamodel.Embedding,
                        datamodel.SubstrateState,
                        datamodel.NodeEmbedding,
                        datamodel.EdgeEmbedding,
                        datamodel.EdgeEmbeddingPart,
                        datamodel.NodeState,
                        datamodel.EdgeState
                    ],
                    safe=True
                )
            except Exception as e:
                raise e
            try:
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
                parameter = {"alpha": 0.5, "beta": 0.5, "timeout": 300}
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

        cls.scenario = concrete_scenarios.VneScenario(
            if_input=datamodel.ConnectionManager(cls.data_source_config),
            if_output=None,
            num_runs=cls.num_runs,
            substrate_generation_settings=cls.substrate_setting.copy(),
            vnr_generation_settings=cls.network_generation_setting.copy(),
            algorithm=cls.algo_setting.copy(),
            arrival_process=cls.ap_setting.copy(),
            service_process=cls.sp_setting.copy(),
            gurobi_settings=cls.gurobi_setting.copy(),
            learning_model_settings=cls.learning_model.copy()
        )
        try:
            cls.scenario.configure_simulations(literals.WRITE_AT_END_STRATEGY)
        except RuntimeError:
            pass

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
                datamodel.RunConfiguration,
                datamodel.RunExecution,
                datamodel.EventOccurrence,
                datamodel.Embedding,
                datamodel.SubstrateState,
                datamodel.NodeEmbedding,
                datamodel.EdgeEmbedding,
                datamodel.EdgeEmbeddingPart,
                datamodel.NodeState,
                datamodel.EdgeState
            ]
        )

    def test_initialization(self):
        simulation = concrete_simulations.VneSimulation(
            sim_strategy=sim_factories.VneStrategyFactory.produce(literals.WRITE_AT_END_STRATEGY),
            substrate_id=self.scenario.simulation_config[0]['substrate_id'],
            event_heap_id=self.scenario.simulation_config[0]['event_heap_id'],
            learning_model_id=-1,
            scenario_id=self.scenario.simulation_config[0]['scenario_id'],
            run_configuration_id=self.scenario.simulation_config[0]['run_configuration_id'],
            algorithm_id=self.scenario.simulation_config[0]['algorithm_id'],
            data_source_config=self.data_source_config.as_dict(),
            data_sink_configs=[self.data_sink_config.as_dict()]
        )
        simulation._initialize()

    def test_run(self):
        simulation = concrete_simulations.VneSimulation(
            sim_strategy=sim_factories.VneStrategyFactory.produce(literals.WRITE_AT_END_STRATEGY),
            substrate_id=self.scenario.simulation_config[0]['substrate_id'],
            event_heap_id=self.scenario.simulation_config[0]['event_heap_id'],
            learning_model_id=-1,
            scenario_id=self.scenario.simulation_config[0]['scenario_id'],
            run_configuration_id=self.scenario.simulation_config[0]['run_configuration_id'],
            algorithm_id=self.scenario.simulation_config[0]['algorithm_id'],
            data_source_config=self.data_source_config.as_dict(),
            data_sink_configs=[self.data_sink_config.as_dict()]
        )
        print 'start with running simulation'
        try:
            simulation.run()
        except AssertionError as e:
            print e
            raise e
        num_embeddings = datamodel.Embedding.select().count()
        self.assertEqual(self.ap_setting['num_requests'], num_embeddings)
        exec_record = datamodel.RunExecution.get(
            datamodel.RunExecution.run_execution_id == simulation.run_execution.identifier
        )
        self.assertEqual(exec_record.stage_of_execution, 2)
        self.assertEqual(
            exec_record.num_successful_embeddings,
            simulation.run_execution.num_successful_embeddings
        )
        self.assertEqual(
            exec_record.num_failed_embeddings,
            simulation.run_execution.num_failed_embeddings
        )
        self.assertEqual(
            exec_record.num_infeasible_embeddings,
            simulation.run_execution.num_infeasible_embeddings
        )
        self.assertEqual(
            exec_record.num_filtered_embeddings,
            simulation.run_execution.num_filtered_embeddings
        )
        self.assertAlmostEqual(
            exec_record.execution_time,
            simulation.run_execution.execution_time,
            delta=0.000001
        )

        num_states = datamodel.SubstrateState.select().count()
        self.assertEqual(
            num_states,
            simulation.run_execution.num_successful_embeddings + self.ap_setting['num_requests']
        )
        print 'Done test for run simulation'

