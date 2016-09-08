import unittest
import algorithms.vne.constants
import algorithms.vne.twostageembedding
import scenario.simulation.vne.helpers
import networkx
import network_model.vne.networkmodel

def find_physical_node (embedding, virtual_node):
    for x in embedding.node_embeddings:
        if x.virtual_node == virtual_node:
            return x.physical_node
    raise Exception


def find_physical_edges (embedding, virtual_edge):
    for x in embedding.edge_embeddings:
        if x.virtual_edge == virtual_edge:
            return x.physical_edges
    raise Exception


def find_cpu_usage(embedding, virtual_node):
    for x in embedding.node_embeddings:
        if x.virtual_node == virtual_node:
            return x.cpu
    raise Exception


def find_bandwidth_usage (embedding, virtual_edge):
    for x in embedding.edge_embeddings:
        if x.virtual_edge == virtual_edge:
            return x.capacity
    raise Exception


class TestVneAlgorithmTwoStage(unittest.TestCase):
    def testEdgeMapping(self):
        edges = list()  # stores the mapping information

        # build networks
        physical_network = network_model.vne.networkmodel.PhysicalNetwork
        virtual_network = network_model.vne.networkmodel.VirtualNetwork

        graph = networkx.Graph()

        graph.add_node(1, cpu=1)
        graph.add_node(2, cpu=1)
        graph.add_node(3, cpu=1)
        graph.add_node(4, cpu=1)
        graph.add_node(5, cpu=1)
        graph.add_node(6, cpu=1)
        graph.add_edge(1, 2, capacity=100)
        graph.add_edge(2, 3, capacity=100)
        graph.add_edge(3, 4, capacity=100)
        graph.add_edge(1, 5, capacity=50)
        graph.add_edge(5, 4, capacity=100)
        graph.add_edge(4, 6, capacity=100)

        physical_network = network_model.vne.networkmodel.PhysicalNetwork(fnss_topology=graph)

        graph = networkx.Graph()

        graph.add_node(1, cpu=1)
        graph.add_node(2, cpu=1)
        graph.add_node(3, cpu=1)
        graph.add_node(4, cpu=1)
        graph.add_edge(1, 2, capacity=30)
        graph.add_edge(1, 3, capacity=70)

        virtual_network = network_model.vne.networkmodel.VirtualNetwork(fnss_topology=graph)

        # create Algorithm and set required information (normally done in the run function)
        algorithm = algorithms.vne.twostageembedding.TwoStageEmbedding(0, 0, physical_network)

        algorithm.vGraph = virtual_network.to_fnss_topology()
        algorithm.virtual = scenario.simulation.vne.helpers.VnettumWrapper(algorithm.vGraph)

        # set Node mapping information
        node_mapping = {1: 1,
                        2: 4,
                        3: 6,
                        4: 2}

        # perform edge mapping
        status_code = algorithm.edgeMapping(edges, node_mapping)

        # check results
        self.assertEqual(status_code, 0, 'No path found')

        for edge in edges:
            v_edge = edge[0]
            p_edge = edge[1]
            if v_edge == [1, 2]:
                self.assertEqual(p_edge, [1, 5, 4], 'Wrong Path for virtual link 1')
            elif v_edge == [1, 3]:
                self.assertEqual(p_edge, [1, 2, 3, 4, 6], 'Wrong Path for virtual link 2')
            else:
                self.assertTrue(True, 'Unexpected virtual link in List (Link %i,%i' % (v_edge[0], v_edge[1]))

        # add new link and check if infeasible is returned

        graph.add_edge(4, 2, capacity=90)

        virtual_network = network_model.vne.networkmodel.VirtualNetwork(fnss_topology=graph)
        algorithm.vGraph = virtual_network.to_fnss_topology()
        algorithm.virtual = scenario.simulation.vne.helpers.VnettumWrapper(algorithm.vGraph)

        status_code = algorithm.edgeMapping(edges, node_mapping)
        self.assertEqual(status_code, 1, 'Successful mapping despite constraint violation')

    def TestWholeProcessGreedy(self):
        # build networks

        pgraph = networkx.Graph()

        pgraph.add_node(1, cpu=100, free_cpu=100)
        pgraph.add_node(2, cpu=0, free_cpu=0)
        pgraph.add_node(3, cpu=0, free_cpu=0)
        pgraph.add_node(4, cpu=50, free_cpu=50)
        pgraph.add_node(5, cpu=0, free_cpu=0)
        pgraph.add_node(6, cpu=30, free_cpu=30)

        pgraph.add_edge(1, 2, capacity=100, free_capacity=100)
        pgraph.add_edge(2, 3, capacity=100, free_capacity=100)
        pgraph.add_edge(3, 4, capacity=100, free_capacity=100)
        pgraph.add_edge(1, 5, capacity=50, free_capacity=50)
        pgraph.add_edge(5, 4, capacity=100, free_capacity=100)
        pgraph.add_edge(4, 6, capacity=100, free_capacity=100)

        physical_network = network_model.vne.networkmodel.PhysicalNetwork(pgraph)

        vgraph = networkx.Graph()

        vgraph.add_node(1, cpu=100)
        vgraph.add_node(2, cpu=50)
        vgraph.add_node(3, cpu=30)
        vgraph.add_edge(1, 2, capacity=30)
        vgraph.add_edge(1, 3, capacity=70)

        virtual_network = network_model.vne.networkmodel.VirtualNetwork(vgraph)

        algorithm = algorithms.vne.twostageembedding.GreedyEmbedding(0, 0, physical_network)
        embedding = algorithm.run(virtual_network, 0)

        self.assertEqual(embedding.vnr_classification, algorithms.vne.constants.embedding_result_success,
                         'Can not embed network')

        # Check if the correct paths are used
        self.assertEqual(find_physical_node(embedding, 1), 1, 'Wrong mapping for Virtual Node 1')
        self.assertEqual(find_physical_node(embedding, 2), 4, 'Wrong mapping for Virtual Node 2')
        self.assertEqual(find_physical_node(embedding, 3), 6, 'Wrong mapping for Virtual Node 3')
        self.assertEqual(find_physical_edges(embedding, (1, 2)), [(1, 5), (5, 4)], 'Wrong mapping for Virtual Link 1')
        self.assertEqual(find_physical_edges(embedding, (1, 3)), [(1, 2), (2, 3), (3, 4), (4, 6)],
                         'Wrong mapping for Virtual Link 2')

        # Check if resources are correct
        self.assertAlmostEqual(find_cpu_usage(embedding, 1), 100, 'Wrong CPU for Node 1')
        self.assertAlmostEqual(find_cpu_usage(embedding, 2), 50, 'Wrong CPU for Node 2')
        self.assertAlmostEqual(find_cpu_usage(embedding, 3), 30, 'Wrong CPU for Node 3')
        self.assertAlmostEqual(find_bandwidth_usage(embedding, (1,2)), 30, 'Wrong BW for Link 1,2')
        self.assertAlmostEqual(find_bandwidth_usage(embedding, (1, 3)), 70, 'Wrong BW for Link 1,3')

        # Try running the embedding a second time and check if it fails this time (resource constraints)
        physical_network.impose_embedding(embedding)
        algorithm = algorithms.vne.twostageembedding.GreedyEmbedding(0, 0, physical_network)
        embedding = algorithm.run(virtual_network, 0)

        self.assertEqual(embedding.vnr_classification, algorithms.vne.constants.embedding_result_infeasible,
                         'Could embed network despite missing resources (or other error occurred)')

    def TestGongGRCCalculation(self):
        # settings
        d = 0.85
        diff = 0.1

        # create network


        graph = networkx.Graph()

        graph.add_node(1, cpu=40)
        graph.add_node(2, cpu=40)
        graph.add_node(3, cpu=200)

        graph.add_edge(1, 2, capacity=250, free_capacity=25)
        graph.add_edge(2, 3, capacity=15, free_capacity=15)

        network = network_model.vne.networkmodel.PhysicalNetwork(graph)
        network.nodes[3].free_cpu = 20
        network.get_edge(1,2).free_capacity = 25

        # calculate GRC
        algorithm = algorithms.vne.twostageembedding.GongEmbedding(alpha=0, beta=0,
                                                                   physicalNetwork=network, gong_d=d,
                                                                   gong_max_diff=diff)

        rating = algorithm.calculate_rating(algorithm.physical, d, diff)

        # compare to pre calculated values
        self.assertAlmostEqual(rating[1], 0.3429781816, None, 'Node 1 wrong GRC')
        self.assertAlmostEqual(rating[2], 0.4572349094, None, 'Node 2 wrong GRC')
        self.assertAlmostEqual(rating[3], 0.199786909, None, 'Node 3 wrong GRC')

        # check correct sorting
        ordered_rating = algorithm.nodeRank(algorithm.physical)
        self.assertEqual(ordered_rating, [2, 1, 3], 'Wrong order')

    def TestWholeProcessGong(self):
        # build networks

        graph = networkx.Graph()

        graph.add_node(1, cpu=100, free_cpu=100)
        graph.add_node(2, cpu=0, free_cpu=0)
        graph.add_node(3, cpu=0, free_cpu=0)
        graph.add_node(4, cpu=50, free_cpu=50)
        graph.add_node(5, cpu=0, free_cpu=0)
        graph.add_node(6, cpu=30, free_cpu=30)

        graph.add_edge(1, 2, capacity=100, free_capacity=100)
        graph.add_edge(2, 3, capacity=70, free_capacity=70)
        graph.add_edge(3, 4, capacity=100, free_capacity=100)
        graph.add_edge(1, 5, capacity=100, free_capacity=100)
        graph.add_edge(5, 4, capacity=100, free_capacity=100)
        graph.add_edge(4, 6, capacity=100, free_capacity=100)

        physical_network = network_model.vne.networkmodel.PhysicalNetwork(graph)

        graph = networkx.Graph()

        graph.add_node(1, cpu=100)
        graph.add_node(2, cpu=50)
        graph.add_node(3, cpu=30)
        graph.add_edge(1, 2, capacity=70)
        graph.add_edge(1, 3, capacity=30)

        virtual_network = network_model.vne.networkmodel.VirtualNetwork(graph)

        algorithm = algorithms.vne.twostageembedding.GongEmbedding(alpha=0, beta=0,
                                                                   physicalNetwork=physical_network, gong_d=0.85,
                                                                   gong_max_diff=0.0001)

        embedding = algorithm.run(virtual_network, 0)

        self.assertEqual(embedding.vnr_classification, algorithms.vne.constants.embedding_result_success,
                         'Can not embed network')

        # Check if the correct paths are used
        self.assertEqual(find_physical_node(embedding, 1), 1, 'Wrong mapping for Virtual Node 1')
        self.assertEqual(find_physical_node(embedding, 2), 4, 'Wrong mapping for Virtual Node 2')
        self.assertEqual(find_physical_node(embedding, 3), 6, 'Wrong mapping for Virtual Node 3')
        self.assertEqual(find_physical_edges(embedding, (1, 2)), [(1, 5), (5, 4)], 'Wrong mapping for Virtual Link 1')
        self.assertEqual(find_physical_edges(embedding, (1, 3)), [(1, 5), (5, 4), (4, 6)],
                         'Wrong mapping for Virtual Link 2')

        # Check if resources are correct
        self.assertEqual(find_cpu_usage(embedding, 1), 100, 'Wrong CPU for Node 1')
        self.assertEqual(find_cpu_usage(embedding, 2), 50, 'Wrong CPU for Node 2')
        self.assertEqual(find_cpu_usage(embedding, 3), 30, 'Wrong CPU for Node 3')

        self.assertEqual(find_bandwidth_usage(embedding, (1, 2)), 70, 'Wrong BW for Link 1,2')
        self.assertEqual(find_bandwidth_usage(embedding, (1,3)), 30, 'Wrong BW for Link 1,3')

        # Try running the embedding a second time
        physical_network.impose_embedding(embedding)

        algorithm = algorithms.vne.twostageembedding.GreedyEmbedding(0, 0, physical_network)
        embedding2 = algorithm.run(virtual_network, 0)

        self.assertEqual(embedding2.vnr_classification, algorithms.vne.constants.embedding_result_infeasible,
                         'Could embed network despite missing resources (or other error occurred)')
