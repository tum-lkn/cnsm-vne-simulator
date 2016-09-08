import numpy as np


class VnettumWrapper(object):
    """
    Interface class for vnettum.algorithms package.

    Does also tracking of stage of life for VNRs and defines some convinience
    functions.
    """

    @classmethod
    def from_object(cls, wrapper):
        return cls(
            network=wrapper.network.copy(),
            substrate=wrapper.substrate
        )

    def __init__(self, network, substrate=False):
        """
        Initializes Object.
        Args:
            network (fnss Topology): Topology created by fnss tool
            substrate (bool, optional): Set to True when network is substrate
        """
        # Depending on whether network is a substrat or not keys for CPU need
        # to be adapted. `cpu` for request results in requsted CPU, `cpu` for
        # substrat in physical available CPU capacity.
        # When `getCPU` for substrate is called, the caller wants the free
        # CPU capacity, though.
        # same goes for the edges.
        self._cpu_key = 'cpu'
        self._capacity_key = 'capacity'
        self.substrate = substrate

        # 0 --> created
        # 1 --> embedded
        # 2 --> departed
        # -1 --> embedding failed
        self.stage_of_life = 0

        if substrate:
            self._cpu_key = 'free_cpu'
            self._capacity_key = 'free_capacity'
        self.network = network
        self.setCPU = self.set_cpu
        self.getCPU = lambda n: self.network.node[n][self._cpu_key]
        self.getMaxCPU = lambda n: self.network.node[n]['cpu']
        self.getMaxBandwidth = lambda n1, n2: self.network.edge[n1][n2]['capacity']
        self.getBandwidth = lambda n1, n2: self.network.edge[n1][n2][self._capacity_key]
        self.setBandwidth = self.set_capacity
        if substrate:
            self.getDistance = lambda n1, n2: np.sqrt(
                np.square(self.network.node[n1]['latitude'] -
                          self.network.node[n2]['latitude']) +
                np.square(self.network.node[n1]['longitude'] -
                          self.network.node[n2]['longitude'])
            )
        else:
            self.getDistance = lambda n1, n2: self.network[n1][n2]['length']

        self.getDelay = lambda n1, n2: self.network.edge[n1][n2]['delay']
        self.getMappedEdges = lambda n1, n2: self.network.edge[n1][n2]['mapping']
        self.getMappedNode = lambda n: self.network.node[n]['mapping']
        self.setMappedNode = self.set_mapped_node
        self.addMappedEdge = self.add_mapped_edge
        self.getPosition = lambda n: (
            self.network.node[n]['longitude'],
            self.network.node[n]['latitude']
        )
        self.incMappedNodes = self.inc_mapped_nodes
        self.incMappedEdges = self.inc_mapped_edges
        # self.node = self.network.node
        # self.edge = self.network.edge

        count = 0
        self.ids_to_edge = {}
        for n, m in network.edges():
            self.ids_to_edge[count] = (n, m)
            count += 1

    def getEdges(self):
        return self.network.edges()

    def getConnectedEdges(self, i):
        return self.network.edges(i)

    def getNodes(self):
        return self.network.nodes()

    def getGraph(self):
        return self.network

    def set_cpu(self, node, cpu):
        """
        Set CPU value of specific node.

        Args:
            node (int): ID of node
            cpu (float): New CPU value

        """
        self.network.node[node][self._cpu_key] = cpu

    def set_capacity(self, node1, node2, capacity):
        """
        Sets new value for capacity of edge.
        Args:
            node1 (int): ID of first endpoint.
            node2 (int): ID of second endpoint.
            capacity (float): New value for remaining capacity.

        """
        self.network.edge[node1][node2][self._capacity_key] = capacity

    def set_mapped_node(self, virtualnode, physicalnode):
        self.network.node[virtualnode]['mapping'] = physicalnode

    def add_mapped_edge(self, vn1, vn2, pn1, pn2):
        self.network.edge[vn1][vn2]['mapping'].append((pn1, pn2))

    def inc_mapped_nodes(self, node):
        self.network.node[node]['currently_mapped'] += 1

    def inc_mapped_edges(self, pn1, pn2):
        self.network.edge[pn1][pn2]['currently_mapped'] += 1

    def dec_mapped_nodes(self, node):
        self.network.node[node]['currently_mapped'] -= 1
        if self.network.node[node]['currently_mapped'] < 0:
            raise ValueError((
                'Negative count for embedded Nodes on physical'
                'node {}'.format(node)
            ))

    def dec_mapped_edges(self, pn1, pn2):
        self.network.edge[pn1][pn2]['currently_mapped'] -= 1
        if self.network.edge[pn1][pn2]['currently_mapped'] < 0:
            raise ValueError((
                'Negative count for embedded Edges on physical'
                'edge ({}-{}('.format(pn1, pn2)
            ))

    def copy(self):
        return VnettumWrapper.from_object(self)
