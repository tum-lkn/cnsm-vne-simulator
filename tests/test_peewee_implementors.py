import logging
import input_output.data_interface.bridge.datamodel as datamodel
import input_output.data_interface.bridge.peewee_implementors as implementors
import input_output.data_interface.interface_config as interface_config
import unittest


__author__ = 'Patrick Kalmbach'


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
                    ],
                    safe=True
                )

                cls.algo_setting = {
                    'name': 'melo_sdp',
                    'parameter': '{"alpha": 0.5, "beta": 0.5, "timeout": 300}'
                }

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
                    'model': 'ER',
                    'connectivity': 0.5,
                    'min_order': 5,
                    'max_order': 15,
                    'min_capacity': 0,
                    'ax_capacity': 50,
                    'min_cpu': 0,
                    'max_cpu': 50,
                    'capacity_generation': 'uniform',
                    'is_substrate': False
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
                    'is_substrate': True
                }
                cls.learning_model = {
                    'type': 'testmode',
                    'serialized_model': '/path/to/model.pkl',
                    'aux_params': 'empty',
                    'comment': 'This model is created from setup method of unittest'
                }
                cls.ap_setting_record = datamodel.ProcessSetting.create(**cls.ap_setting)
                cls.sp_setting_record = datamodel.ProcessSetting.create(**cls.sp_setting)
                network_generation_setting = datamodel.NetworkGenerationSetting.create(
                    **cls.network_generation_setting
                )

                cls.event_generation_setting =  {
                    #'arrival_process_setting': ap_setting.process_setting_id,
                    #'service_process_Setting': sp_setting.process_setting_id,
                    'network_setting': network_generation_setting.network_generation_setting_id
                }

                event_generation_setting = datamodel.EventGenerationSetting.create(
                    **cls.event_generation_setting
                )
                datamodel.ProcessToEventSetting.create(
                    event_generation_setting=event_generation_setting,
                    process_setting=cls.ap_setting_record
                )
                datamodel.ProcessToEventSetting.create(
                    event_generation_setting=event_generation_setting,
                    process_setting=cls.sp_setting_record
                )

                substrate_setting = datamodel.NetworkGenerationSetting.create(
                    **cls.substrate_setting
                )

                learning_model = datamodel.LearningModel.create(
                    **cls.learning_model
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
        cls.connection_params = {
            'database': 'TestDb',
            'host': '127.0.0.1',
            'port': 3306,
            'user': 'root',
            'passwd': 'root'
        }
        cls.input_if = datamodel.ConnectionManager(interface_config.PeeweeInterfaceConfig(**cls.connection_params))
        cls.input_if.connect()

        cls._num_runs = 2
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
            ]
        )


class StochasticProcessImplementorTest(AbstractTest):

    def test_get_object_id(self):
        implementor = implementors.StochasticProcessImplementor(self.input_if)
        identifier = implementor.get_object_id(
            **self.ap_setting
        )
        assert identifier == self.ap_setting_record.process_setting_id
        identifier = implementor.get_object_id(
            **self.sp_setting
        )
        assert identifier == self.sp_setting_record.process_setting_id


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(StochasticProcessImplementorTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
