import unittest
import algorithms.vne.constants
import algorithms.vne.modifiedmelo
import scenario.simulation.vne.helpers
import networkx


class TestVNEAlgorithmsMelo(unittest.TestCase):
    def testSDPbasic(self):
        physical_network = scenario.simulation.vne.helpers.VnettumWrapper(networkx.Graph(), True)
        virtual_network = scenario.simulation.vne.helpers.VnettumWrapper(networkx.Graph(), False)

        graph = physical_network.getGraph()

        graph.add_node(1, cpu=100, free_cpu=100)
        graph.add_node(2, cpu=0, free_cpu=0)
        graph.add_node(3, cpu=0, free_cpu=0)
        graph.add_node(4, cpu=50, free_cpu=50)
        graph.add_node(5, cpu=0, free_cpu=0)
        graph.add_node(6, cpu=30, free_cpu=30)

        # fake geo positions
        for n in graph.nodes():
            graph.node[n]['latitude'] = 0
            graph.node[n]['longitude'] = 0

        graph.add_edge(1, 2, capacity=100, free_capacity=100)
        graph.add_edge(2, 3, capacity=100, free_capacity=100)
        graph.add_edge(3, 4, capacity=100, free_capacity=100)
        graph.add_edge(1, 5, capacity=50, free_capacity=50)
        graph.add_edge(5, 4, capacity=100, free_capacity=100)
        graph.add_edge(4, 6, capacity=100, free_capacity=100)

        for (i, j) in graph.edges():
            graph.edge[i][j]['delay'] = 0
        graph = virtual_network.getGraph()

        graph.add_node(1, cpu=100)
        graph.add_node(2, cpu=50)
        graph.add_node(3, cpu=30)
        graph.add_edge(1, 2, capacity=30)
        graph.add_edge(1, 3, capacity=75)

        for (i, j) in graph.edges():
            graph.edge[i][j]['length'] = 0
            graph.edge[i][j]['delay'] = 999

        for (i, j) in graph.edges():
            graph.edge[i][j]['mapping'] = list()
        algorithm = algorithms.vne.modifiedmelo.MeloSDP(1, 1, physical_network)

        status_code = algorithm.run(virtual_network, None, 0)

        self.assertEqual(status_code['success'], algorithms.vne.constants.embedding_result_success,
                         'Can not embed network')

        # Check if the correct paths are used
        self.assertEqual(virtual_network.getMappedNode(1), 1, 'Wrong mapping for Virtual Node 1')
        self.assertEqual(virtual_network.getMappedNode(2), 4, 'Wrong mapping for Virtual Node 2')
        self.assertEqual(virtual_network.getMappedNode(3), 6, 'Wrong mapping for Virtual Node 3')
        self.assertEqual(virtual_network.getMappedEdges(1, 2), [(1, 5), (4, 5)], 'Wrong mapping for Virtual Link 1')
        self.assertEqual(virtual_network.getMappedEdges(1, 3), [(1, 2), (2, 3), (3, 4), (4, 6)],
                         'Wrong mapping for Virtual Link 2')

        # Check if residual resources are correct
        self.assertEqual(physical_network.getCPU(1), 0, 'Wrong CPU for Node 1')
        self.assertEqual(physical_network.getCPU(2), 0, 'Wrong CPU for Node 2')
        self.assertEqual(physical_network.getCPU(3), 0, 'Wrong CPU for Node 3')
        self.assertEqual(physical_network.getCPU(4), 0, 'Wrong CPU for Node 4')
        self.assertEqual(physical_network.getCPU(5), 0, 'Wrong CPU for Node 5')
        self.assertEqual(physical_network.getCPU(6), 0, 'Wrong CPU for Node 6')
        self.assertEqual(physical_network.getBandwidth(1, 2), 25, 'Wrong BW for Link 1,2')
        self.assertEqual(physical_network.getBandwidth(2, 3), 25, 'Wrong BW for Link 2,3')
        self.assertEqual(physical_network.getBandwidth(3, 4), 25, 'Wrong BW for Link 3,4')
        self.assertEqual(physical_network.getBandwidth(1, 5), 20, 'Wrong BW for Link 1,2')
        self.assertEqual(physical_network.getBandwidth(5, 4), 70, 'Wrong BW for Link 5,4')
        self.assertEqual(physical_network.getBandwidth(4, 6), 25, 'Wrong BW for Link 4,6')

        # Try running the embedding a second time and check if it fails this time (resource constraints)
        status_code = algorithm.run(virtual_network, None, 0)

        self.assertEqual(status_code['success'], algorithms.vne.constants.embedding_result_infeasible,
                         'Could embed network despite missing resources (or other error occurred)')


    def testLBbasic(self):
        physical_network = scenario.simulation.vne.helpers.VnettumWrapper(networkx.Graph(), True)
        virtual_network = scenario.simulation.vne.helpers.VnettumWrapper(networkx.Graph(), False)

        graph = physical_network.getGraph()

        graph.add_node(1, cpu=100, free_cpu=100)
        graph.add_node(2, cpu=1, free_cpu=0)
        graph.add_node(3, cpu=1, free_cpu=0)
        graph.add_node(4, cpu=50, free_cpu=50)
        graph.add_node(5, cpu=1, free_cpu=0)
        graph.add_node(6, cpu=30, free_cpu=30)

        # fake geo positions
        for n in graph.nodes():
            graph.node[n]['latitude'] = 0
            graph.node[n]['longitude'] = 0

        graph.add_edge(1, 2, capacity=100, free_capacity=100)
        graph.add_edge(2, 3, capacity=100, free_capacity=100)
        graph.add_edge(3, 4, capacity=100, free_capacity=100)
        graph.add_edge(1, 5, capacity=50, free_capacity=50)
        graph.add_edge(5, 4, capacity=100, free_capacity=100)
        graph.add_edge(4, 6, capacity=100, free_capacity=100)

        for (i, j) in graph.edges():
            graph.edge[i][j]['delay'] = 0
        graph = virtual_network.getGraph()

        graph.add_node(1, cpu=100)
        graph.add_node(2, cpu=50)
        graph.add_node(3, cpu=30)
        graph.add_edge(1, 2, capacity=30)
        graph.add_edge(1, 3, capacity=75)

        for (i, j) in graph.edges():
            graph.edge[i][j]['length'] = 0
            graph.edge[i][j]['delay'] = 999

        for (i, j) in graph.edges():
            graph.edge[i][j]['mapping'] = list()
        algorithm = algorithms.vne.modifiedmelo.MeloLB(1, 1, physical_network)

        status_code = algorithm.run(virtual_network, None, 0)

        self.assertEqual(status_code['success'], algorithms.vne.constants.embedding_result_success,
                         'Can not embed network')

        # Check if the correct paths are used
        self.assertEqual(virtual_network.getMappedNode(1), 1, 'Wrong mapping for Virtual Node 1')
        self.assertEqual(virtual_network.getMappedNode(2), 4, 'Wrong mapping for Virtual Node 2')
        self.assertEqual(virtual_network.getMappedNode(3), 6, 'Wrong mapping for Virtual Node 3')
        self.assertEqual(virtual_network.getMappedEdges(1, 2), [(1, 5), (4, 5)], 'Wrong mapping for Virtual Link 1')
        self.assertEqual(virtual_network.getMappedEdges(1, 3), [(1, 2), (2, 3), (3, 4), (4, 6)],
                         'Wrong mapping for Virtual Link 2')

        # Check if residual resources are correct
        self.assertEqual(physical_network.getCPU(1), 0, 'Wrong CPU for Node 1')
        self.assertEqual(physical_network.getCPU(2), 0, 'Wrong CPU for Node 2')
        self.assertEqual(physical_network.getCPU(3), 0, 'Wrong CPU for Node 3')
        self.assertEqual(physical_network.getCPU(4), 0, 'Wrong CPU for Node 4')
        self.assertEqual(physical_network.getCPU(5), 0, 'Wrong CPU for Node 5')
        self.assertEqual(physical_network.getCPU(6), 0, 'Wrong CPU for Node 6')
        self.assertEqual(physical_network.getBandwidth(1, 2), 25, 'Wrong BW for Link 1,2')
        self.assertEqual(physical_network.getBandwidth(2, 3), 25, 'Wrong BW for Link 2,3')
        self.assertEqual(physical_network.getBandwidth(3, 4), 25, 'Wrong BW for Link 3,4')
        self.assertEqual(physical_network.getBandwidth(1, 5), 20, 'Wrong BW for Link 1,2')
        self.assertEqual(physical_network.getBandwidth(5, 4), 70, 'Wrong BW for Link 5,4')
        self.assertEqual(physical_network.getBandwidth(4, 6), 25, 'Wrong BW for Link 4,6')

        # Try running the embedding a second time and check if it fails this time (resource constraints)
        status_code = algorithm.run(virtual_network, None, 0)

        self.assertEqual(status_code['success'], algorithms.vne.constants.embedding_result_infeasible,
                         'Could embed network despite missing resources (or other error occurred)')


    def testWSDPbasic(self):
        physical_network = scenario.simulation.vne.helpers.VnettumWrapper(networkx.Graph(), True)
        virtual_network = scenario.simulation.vne.helpers.VnettumWrapper(networkx.Graph(), False)

        graph = physical_network.getGraph()

        graph.add_node(1, cpu=100, free_cpu=100)
        graph.add_node(2, cpu=1, free_cpu=0)
        graph.add_node(3, cpu=1, free_cpu=0)
        graph.add_node(4, cpu=50, free_cpu=50)
        graph.add_node(5, cpu=1, free_cpu=0)
        graph.add_node(6, cpu=30, free_cpu=30)

        # fake geo positions
        for n in graph.nodes():
            graph.node[n]['latitude'] = 0
            graph.node[n]['longitude'] = 0

        graph.add_edge(1, 2, capacity=100, free_capacity=100)
        graph.add_edge(2, 3, capacity=100, free_capacity=100)
        graph.add_edge(3, 4, capacity=100, free_capacity=100)
        graph.add_edge(1, 5, capacity=50, free_capacity=50)
        graph.add_edge(5, 4, capacity=100, free_capacity=100)
        graph.add_edge(4, 6, capacity=100, free_capacity=100)

        for (i, j) in graph.edges():
            graph.edge[i][j]['delay'] = 0
        graph = virtual_network.getGraph()

        graph.add_node(1, cpu=100)
        graph.add_node(2, cpu=50)
        graph.add_node(3, cpu=30)
        graph.add_edge(1, 2, capacity=30)
        graph.add_edge(1, 3, capacity=75)

        for (i, j) in graph.edges():
            graph.edge[i][j]['length'] = 0
            graph.edge[i][j]['delay'] = 999

        for (i, j) in graph.edges():
            graph.edge[i][j]['mapping'] = list()
        algorithm = algorithms.vne.modifiedmelo.MeloWSDP(1, 1, physical_network)

        status_code = algorithm.run(virtual_network, None, 0)

        self.assertEqual(status_code['success'], algorithms.vne.constants.embedding_result_success,
                         'Can not embed network')

        # Check if the correct paths are used
        self.assertEqual(virtual_network.getMappedNode(1), 1, 'Wrong mapping for Virtual Node 1')
        self.assertEqual(virtual_network.getMappedNode(2), 4, 'Wrong mapping for Virtual Node 2')
        self.assertEqual(virtual_network.getMappedNode(3), 6, 'Wrong mapping for Virtual Node 3')
        self.assertEqual(virtual_network.getMappedEdges(1, 2), [(1, 5), (4, 5)], 'Wrong mapping for Virtual Link 1')
        self.assertEqual(virtual_network.getMappedEdges(1, 3), [(1, 2), (2, 3), (3, 4), (4, 6)],
                         'Wrong mapping for Virtual Link 2')

        # Check if residual resources are correct
        self.assertEqual(physical_network.getCPU(1), 0, 'Wrong CPU for Node 1')
        self.assertEqual(physical_network.getCPU(2), 0, 'Wrong CPU for Node 2')
        self.assertEqual(physical_network.getCPU(3), 0, 'Wrong CPU for Node 3')
        self.assertEqual(physical_network.getCPU(4), 0, 'Wrong CPU for Node 4')
        self.assertEqual(physical_network.getCPU(5), 0, 'Wrong CPU for Node 5')
        self.assertEqual(physical_network.getCPU(6), 0, 'Wrong CPU for Node 6')
        self.assertEqual(physical_network.getBandwidth(1, 2), 25, 'Wrong BW for Link 1,2')
        self.assertEqual(physical_network.getBandwidth(2, 3), 25, 'Wrong BW for Link 2,3')
        self.assertEqual(physical_network.getBandwidth(3, 4), 25, 'Wrong BW for Link 3,4')
        self.assertEqual(physical_network.getBandwidth(1, 5), 20, 'Wrong BW for Link 1,2')
        self.assertEqual(physical_network.getBandwidth(5, 4), 70, 'Wrong BW for Link 5,4')
        self.assertEqual(physical_network.getBandwidth(4, 6), 25, 'Wrong BW for Link 4,6')

        # Try running the embedding a second time and check if it fails this time (resource constraints)
        status_code = algorithm.run(virtual_network, None, 0)

        self.assertEqual(status_code['success'], algorithms.vne.constants.embedding_result_infeasible,
                         'Could embed network despite missing resources (or other error occurred)')