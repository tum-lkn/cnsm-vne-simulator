import unittest
import os

import input_output.data_interface.interface_config as interface_config
import logging
import input_output.data_interface.bridge.datamodel as datamodel
import control.control as ctrl
import literals
import sys
from input_output.data_interface.bridge.abstract_implementor import AbstractImplementor


class SystemTest(unittest.TestCase):
    data_source_config = interface_config.PeeweeInterfaceConfig(
        database='TestDb',
        host='127.0.0.1',
        port=3306,
        user='root',
        passwd='root'
    )

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

    def test_system(self):
        """
        Note:
            You may have to adapt the vne_system_test.ini in configfiles
            filter and no filter test cannot be run on the same config file!

        """
        sys.argv.pop(1)
        # Nosetest adds the test it executes to the sys.argv, remove it or
        # else control crashes
        control = ctrl.Control(
            configurationfile=os.path.join(
                os.path.dirname(__file__),
                'configfiles',
                'vne_system_test.ini'
            )
        )
        control.start()
        self.assertEqual(
            datamodel.NetworkGenerationToNetwork.select().count(), 4
        )
        self.assertEqual(
            datamodel.Scenario.select().count(), 2
        )
        self.assertEqual(
            datamodel.RunConfiguration.select().count(), 4
        )
        self.assertEqual(
            datamodel.RunExecution.select().count(), 4
        )
        self.assertEqual(
            datamodel.Network.select().where(datamodel.Network.subclass_type == literals.CLASS_PHYSICAL_NETWORK).count(),
            4
        )
        self.assertEqual(datamodel.AlgorithmSetting.select().count(), 1)
        self.assertEqual(datamodel.ProcessSetting.select().count(), 2)
        self.assertEqual(datamodel.EventGenerationSetting.select().count(), 1)
        self.assertEqual(datamodel.ProcessToEventSetting.select().count(), 2)
        print 'System test finished'

    def test_system_with_filter(self):
        """
        Note:
            You may have to adapt the vne_system_test.ini in configfiles
            filter and no filter test cannot be run on the same config file!

        """
        sys.argv.pop(1)
        aux_params = {}
        aux_params['aux_params'] = {"output_transfer": "sigmoid", "loss": "bern_ces", "din": 36, "dout": 1,
                     "hlayers": [200], "std": [276.27385142857145, 92.18768,
                     0.295015214980061, 0.3631751462266762, 0.704984785019939,
                     0.6367967254413526, 8534.136525714286, 3488.8932114285712,
                     3.9736228571428573, 0.06870467349371139, 4.6791124927119006,
                     5.977485714285714, 3.7552, 0.6059714181064139, 0.134753835108421,
                     0.056255779895266846, 49.09024, 112.61675428571428, 4.216223542872942,
                     -2.894577396290225, 49.08989714285714, 3.9641405439939077,
                     0.8587628969263279, -129.8747618294985, 2.066026733747659,
                     0.10348107556069695, 0.6266068257683556, 0.09412178921378062,
                     124.03708223833604, 9.347017142857142, 121.85962774692571,
                     2.678280928287532, 3.8316342857142858, 9.340022857142857,
                     0.042615318823466156, 21.76272], "hidden_transfer": ["tanh"],
                     "optimizer": ["rmsprop", {"momentum": 0.7, "step_rate": 0.001,
                     "decay": 0.9}], "mean": [276.27385142857145, 92.18768,
                     0.295015214980061, 0.3631751462266762, 0.704984785019939,
                     0.6367967254413526, 8534.136525714286, 3488.8932114285712,
                     3.9736228571428573, 0.06870467349371139, 4.6791124927119006,
                     5.977485714285714, 3.7552, 0.6059714181064139, 0.134753835108421,
                     0.056255779895266846, 49.09024, 112.61675428571428,
                     4.216223542872942, -2.894577396290225, 49.08989714285714,
                     3.9641405439939077, 0.8587628969263279, -129.8747618294985,
                     2.066026733747659, 0.10348107556069695, 0.6266068257683556,
                     0.09412178921378062, 124.03708223833604, 9.347017142857142,
                     121.85962774692571, 2.678280928287532, 3.8316342857142858,
                     9.340022857142857, 0.042615318823466156, 21.76272]}
        aux_params['successor_name'] = 'SDP'
        aux_params['serialized_model'] = '/home/patrick/Documents/GitHub/lkn/deep-sdn/models/breze_rnn_eo50_eorq.pkl'
        aux_params['timeout'] = 10
        aux_params['alpha'] = 0.5
        aux_params['beta'] = 0.5
        aux_params['type'] = 'breze_rnn'

        connection = datamodel.ConnectionManager(input_config=self.data_source_config)
        connection.connect()
        datamodel.AlgorithmSetting.create(
            name='RNN_FILTER',
            parameter=AbstractImplementor(None).serialize(aux_params)
        )
        connection.disconnect()

        control = ctrl.Control(
            configurationfile=os.path.join(
                os.path.dirname(__file__),
                'configfiles',
                'vne_system_test.ini'
            )
        )
        control.start()
        self.assertEqual(
            datamodel.NetworkGenerationToNetwork.select().count(), 4
        )
        self.assertEqual(
            datamodel.Scenario.select().count(), 2
        )
        self.assertEqual(
            datamodel.RunConfiguration.select().count(), 4
        )
        self.assertEqual(
            datamodel.RunExecution.select().count(), 4
        )
        self.assertEqual(
            datamodel.Network.select().where(datamodel.Network.subclass_type == literals.CLASS_PHYSICAL_NETWORK).count(),
            4
        )
        self.assertEqual(datamodel.AlgorithmSetting.select().count(), 1)
        self.assertEqual(datamodel.ProcessSetting.select().count(), 2)
        self.assertEqual(datamodel.EventGenerationSetting.select().count(), 1)
        self.assertEqual(datamodel.ProcessToEventSetting.select().count(), 2)
        print 'System test finished'

if __name__ == '__main__':
    unittest.main()
