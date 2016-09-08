import unittest
import input_output.data_interface.vne.bridge.datamodel as vne_interface
import fnss
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)


class DbManagerTest(unittest.TestCase):
    def setUp(self):
        self.connection_params = {
            'db_name': 'TestDb',
            'host': 'localhost',
            'port': 3366,
            'user': 'root',
            'password': 'root'
        }
        vne_interface.DbManager.set_connection_parameter(**self.connection_params)
        vne_interface.DbManager.connect()
        with vne_interface.dm.db.transaction() as txn:
            try:
                vne_interface.dm.db.create_tables([
                    vne_interface.dm.EventGenerationSetting,
                    vne_interface.dm.VnrGenerationSetting,
                    vne_interface.dm.AlgorithmSetting,
                    vne_interface.dm.Network,
                    vne_interface.dm.Node,
                    vne_interface.dm.Edge,
                    vne_interface.dm.NetworkAttribute,
                    vne_interface.dm.SubstrateGenerationSetting,
                    vne_interface.dm.EventHeap,
                    vne_interface.dm.Experiment,
                    vne_interface.dm.LearningModel,
                    vne_interface.dm.Setup
                ])

                self.algo_setting = vne_interface.dm.AlgorithmSetting.create(
                    parameter='{"alpha": 0.5, "beta": 0.5, "name": "melo_sdp", "timeout": 65.0}'
                )
                self.event_setting = vne_interface.dm.EventGenerationSetting.create(
                    parameter='{"model": "erdoes_renyi", "lmbd": 10, "num_requests": 5, "avg_life_time": 1000}'
                )
                self.vnr_setting = vne_interface.dm.VnrGenerationSetting.create(
                    parameter='{"min_bw": 0, "max_cpu": 50, "maxy": 100, "miny": 0, '
                              '"min_cpu": 0, "max_bw": 50, "minx": 0, "connectivity": 0.5, '
                              '"bandwidth": "uniform", "max_num_nodes": 22, '
                              '"min_num_nodes": 8, "max_distance": 501, '
                              '"min_distance": 500, "maxx": 100}'
                )
                self.substrate_setting = vne_interface.dm.SubstrateGenerationSetting.create(
                    parameter='{"min_bw": 50, "max_cpu": 100, "maxy": 100, '
                              '"substrate": true, "miny": 0, "min_cpu": 50, '
                              '"bandwidth": "uniform", "num_nodes": 43, "minx": 0, '
                              '"connectivity": 0.5, "max_bw": 100, "max_distance": 50, '
                              '"model": "erdoes_renyi", "min_distance": 1, "maxx": 100}'
                )
                substrate = fnss.erdos_renyi_topology(43, 0.5)
                fnss.set_capacities_random_uniform(substrate, range(50, 100))
                for attributes in substrate.node.itervalues():
                    attributes['cpu'] = np.random.uniform(50, 100)
                networkif = vne_interface.Network.from_fnss_topology(substrate)
                self.substrate = networkif.network_model
                self.learning_model = vne_interface.dm.LearningModel.create(
                    type='testmode',
                    serialized_model='/path/to/model.pkl',
                    aux_params='empty',
                    comment='This model is created from setup method of unittest'
                )
                self.experiment = vne_interface.dm.Experiment.create(
                    description='Created for test purposes from setup method of unittest'
                )
                self.event_heaps = []
                for i in range(5):
                    self.event_heaps.append(
                        vne_interface.dm.EventHeap.create(
                            event_generation_setting_id=self.event_setting.event_generation_setting_id,
                            vnr_generation_setting_id=self.vnr_setting.vnr_generation_setting_id
                        )
                    )
                self.setups = []
                for heap in self.event_heaps:
                    self.setups.append(
                        vne_interface.dm.Setup.create(
                            event_generation_setting=self.event_setting,
                            vnr_generation_setting=self.vnr_setting,
                            substrate_generation_setting=self.substrate_setting,
                            algorithm_setting=self.algo_setting,
                            experiment=self.experiment,
                            event_heap=heap,
                            description='Test Setup created from unittest setup method',
                            substrate=self.substrate,
                            learning_model_id=self.learning_model.learning_model_id
                        )
                    )
                txn.commit()
            except Exception as e:
                logging.exception(e)
                txn.rollback()

    def tearDown(self):
        pass
        vne_interface.dm.db.drop_tables([
            vne_interface.dm.EventGenerationSetting,
            vne_interface.dm.VnrGenerationSetting,
            vne_interface.dm.AlgorithmSetting,
            vne_interface.dm.Network,
            vne_interface.dm.Node,
            vne_interface.dm.Edge,
            vne_interface.dm.NetworkAttribute,
            vne_interface.dm.SubstrateGenerationSetting,
            vne_interface.dm.EventHeap,
            vne_interface.dm.Experiment,
            vne_interface.dm.LearningModel,
            vne_interface.dm.Setup
        ])

    def test_get_algo_id(self):
        algoid = vne_interface.DbManager.get_algo_id('melo_sdp', 65)
        self.assertEqual(
            algoid,
            self.algo_setting.algorithm_setting_id,
            'Algorithm ids do not match. Got {} expected {}'.format(
                algoid,
                self.algo_setting.algorithm_setting_id
            )
        )

    def test_get_substrate_id(self):
        substrate_id = vne_interface.DbManager.get_substrate_id(
            order=43,
            model='ER'
        )
        self.assertEqual(
            substrate_id,
            self.substrate.network_id,
            'Substrate ids do not match. Got {} expected {}'.format(
                substrate_id,
                self.substrate.network_id
            )
        )

    def test_get_event_heap_ids(self):
        event_heap_ids = vne_interface.DbManager.get_event_heap_ids(
            min_order=8,
            max_order=22,
            model='ER',
            arrival_rate=10
        )
        self.assertEqual(
            len(event_heap_ids),
            len(self.event_heaps),
            'Number of found event heaps not correct. Expected {} found {}'.format(
                len(self.event_heaps),
                len(event_heap_ids)
            )
        )
        self.assertListEqual(
            event_heap_ids,
            [heap.event_heap_id for heap in self.event_heaps],
            (
                'Event Heap Ids are note the same. Expected list with '
                'elements {} got list {}'.format(
                    [heap.event_heap_id for heap in self.event_heaps],
                    event_heap_ids
                )
            )
        )

    def test_get_learning_model_id(self):
        learning_model_id = vne_interface.DbManager.get_learning_model_id(
            path=self.learning_model.serialized_model
        )
        self.assertEqual(
            learning_model_id,
            self.learning_model.learning_model_id,
            'Learning model ids not the same, found {} expected {}'.format(
                learning_model_id,
                self.learning_model.learning_model_id
            )
        )
