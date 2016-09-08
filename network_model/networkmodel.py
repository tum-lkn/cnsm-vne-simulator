""" Thid module contains a network model for VNE simulations.
"""
import fnss
import networkx as nx
import numpy as np
import literals
import logging

__authors__ = 'Patrick Kalmbach'


class Node(object):

    def __init__(self, label, cpu=None, latitude=None, longitude=None):
        """ Initializes Object.

            Args:
                label (int): Identifier of Node.
                cpu (float): CPU capacity of node.
                latitude (float, optional): Latitute coordinate of node.
                longitude (float, optional): Longitude coordinate of node.

            Remark:
                The `label` attribute is not the actual identifier of a data source
                of the node but the identifier relative to the network (as in
                node number 5 vs. node_id 501 4587 202).
        """
        self.label = label
        self._cpu = cpu
        self.latitude =latitude
        self.longitude = longitude

    @property
    def cpu(self):
        return self._cpu

    @cpu.setter
    def cpu(self, cpu):
        assert cpu >= 0, 'occupied cpu too small, new value {}'.format(cpu)
        self._cpu = cpu

    def todict(self):
        """ Creates dictionary of attributes.

            Returns:
                dictionary: Dictionary containing objects attributes
        """
        dictionary = {}
        for key, value in self.__dict__.iteritems():
            if key.startswith('_'):
                key = key[1:]
            dictionary[key] = value
        return dictionary


class VirtualNode(Node):

    def __init__(self, label, cpu=None, latitude=None, longitude=None):
        """ Initializes Object.

            Args:
                label (int): Identifier of Node.
                cpu (float): CPU capacity of node.
                latitute (float, optional): Latitute coordinate of node.
                longitude (float, optional): Longitude coordinate of node.
        """
        super(VirtualNode, self).__init__(label, cpu, latitude, longitude)
        self.mapped_to = None


class PhysicalNode(Node):

    def __init__(self, label, cpu=None, latitude=None, longitude=None):
        """ Initializes Object.

            Args:
                label (int): Identifier of Node.
                cpu (float): CPU capacity of node.
                latitute (float, optional): Latitute coordinate of node.
                longitude (float, optional): Longitude coordinate of node.
        """
        super(PhysicalNode, self).__init__(label, cpu, latitude, longitude)
        self.mapping = None
        self.total_mapped = 0
        self._currently_mapped = 0
        self._occupied_cpu = 0
        self._free_cpu = self.cpu

    @property
    def currently_mapped(self):
        return self._currently_mapped

    @currently_mapped.setter
    def currently_mapped(self, mapped):
        assert mapped >= 0
        self._currently_mapped = mapped

    @property
    def occupied_cpu(self):
        return self._occupied_cpu

    @occupied_cpu.setter
    def occupied_cpu(self, cpu):
        assert \
            cpu > -0.001, \
            'occupied cpu too small, new value {}'.format(cpu)
        assert \
            cpu < self.cpu + 0.001, \
            'occupied cpu too large, total cpu {}, new value {}'.format(
                self.cpu,
                cpu
            )
        self._occupied_cpu = cpu

    @property
    def free_cpu(self):
        return self._free_cpu

    @free_cpu.setter
    def free_cpu(self, cpu):
        assert \
            cpu > -0.001, \
            'free cpu too small, new value {}'.format(cpu)
        assert \
            cpu < self.cpu + 0.001, \
            'free cpu too large, total cpu {}, new value {}'.format(
                self.cpu,
                cpu
            )
        self._free_cpu = cpu


class Edge(object):

    def __init__(self, label, node_one_id, node_two_id, capacity=None, delay=None,
                 length=None):
        """ Initializes Object.

            Args:
                id (int): Identifier of Edge.
                node_one_id (int): Identifier of first incident node.
                node_two_id (int): Identifier of second incident node.
                capacity (float): Capacity of Edge.
                delay (float): Delay of edge in ms.
                length (float): length of edge in Km.
        """
        self.node_one_id = node_one_id
        self.node_two_id = node_two_id
        self.label = label
        self._capacity = capacity
        self._delay = delay
        self._length = length

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, cap):
        assert cap >= 0, 'capacity too small, new value {}'.format(cap)
        self._capacity = cap

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, delay):
        assert delay >= 0
        self._delay = delay

    @property
    def length(self):
        return self.length

    @length.setter
    def length(self, length):
        assert length >= 0
        self._length = length

    def todict(self):
        """ Creates dictionary of attributes.

            Returns:
                dictionary: Dictionary containing objects attributes
        """
        dictionary = {}
        for key, value in self.__dict__.iteritems():
            if key.startswith('_'):
                key = key[1:]
            dictionary[key] = value
        return dictionary


class VirtualEdge(Edge):

    def __init__(self, label, node_one_id, node_two_id, capacity=None, delay=None, length=None):
        """ Initializes Object.

            Args:
                label (int): Identifier of Edge.
                node_one_id (int): Identifier of first incident node.
                node_two_id (int): Identifier of second incident node.
                capacity (float): Capacity of Edge.
                delay (float): Delay of edge in ms.
                length (float): length of edge in Km.
        """
        super(VirtualEdge, self).__init__(label,  node_one_id, node_two_id,
                                          capacity, delay, length)


class PhysicalEdge(Edge):

    def __init__(self, label, node_one_id, node_two_id, capacity=None, delay=None, length=None):
        """ Initializes Object.

            Args:
                label (int): Identifier of Edge.
                node_one_id (int): Identifier of first incident node.
                node_two_id (int): Identifier of second incident node.
                capacity (float): Capacity of Edge.
                delay (float): Delay of edge in ms.
                length (float): length of edge in Km.
        """
        super(PhysicalEdge, self).__init__(label, node_one_id, node_two_id,
                                           capacity, delay, length)
        self._free_capacity = self.capacity
        self._occupied_capacity = 0.
        self._currently_mapped = 0
        self.total_mapped = 0

    @property
    def free_capacity(self):
        return self._free_capacity

    @free_capacity.setter
    def free_capacity(self, capacity):
        assert \
            capacity > -0.001, \
            'free capacity too small, new value {}'.format(capacity)
        assert \
            capacity < self.capacity + 0.001, \
            'free capacity too large, total capacity {}, new value {}'.format(
                self.capacity,
                capacity
            )
        self._free_capacity = capacity

    @property
    def occupied_capacity(self):
        return self._occupied_capacity

    @occupied_capacity.setter
    def occupied_capacity(self, capacity):
        assert \
            capacity > -0.001, \
            'occupied capacity too small, new value {}'.format(capacity)
        assert \
            capacity < self.capacity + 0.001, \
            'occupied capacity too large, total capacity {}, new value {}'.format(
                self.capacity,
                capacity
            )
        self._occupied_capacity = capacity

    @property
    def currently_mapped(self):
        return self._currently_mapped

    @currently_mapped.setter
    def currently_mapped(self, count):
        assert count >= 0
        self._currently_mapped = count


class Network(object):

    @classmethod
    def from_datasource(cls, implementor, object_id):
        """ Retrieves object from data source given an identifier.

            Args:
                implementor (input_output.vne.bridge.abstract_implementor.AbstractImplementor):
                    Some object specifying sink. For more details refer to class
                    ObjectFactory in peewee_implementors.py
                identifier (object_id): Some unique identifier
        """
        object = implementor.get_object(object_id)
        object.implementor = implementor
        return object

    @classmethod
    def get_identifier(cls, implementor, **kwargs):
        """ Retrieves identifier from data source

            Args:
                implementor (input_output.vne.bridge.abstract_implementor.AbstractImplementor):
                    Some object specifying sink. For more details refer to class
                    ObjectFactory in peewee_implementors.py
        """
        return implementor.get_object_id(**kwargs)

    def __init__(self, fnss_topology=None, node_type=Node, edge_type=Edge,
                 identifier=None, implementor=None, model=None):
        """ Initializes Object.

            Args:
                fnss_topology (fnss.Topology): Topology to create network from.
                node_type (class): Class of Nodes for network.
                edge_type (class): Class of Edges for network.
                identifier (object): Unique identifier.
                implementor (input_output.data_interface.vne.bridge.abstract_implementor.AbstractImplementor):
                    Derivation of `AbstractImplementor` handling input output
                    to data source.
                model (string): Model of network.
        """
        self.logger = logging.getLogger(str(self.__class__))
        self.logger.setLevel(level=logging.DEBUG)
        self.implementor = implementor
        self.average_neighbour_degree = 0
        self.std_neighbour_degree = 0
        self.average_clustering_coefficient = 0
        self.std_clustering_coefficient = 0
        self.average_effective_eccentricity = 0
        self.std_effective_eccentricity = 0
        self.max_effective_eccentricity = 0
        self.min_effective_eccentricity = 0
        self.average_path_length = 0
        self.std_path_length = 0
        self.percentage_central_points = 0
        self.percentage_end_points = 0
        self.num_nodes = 0
        self.num_edges = 0
        self.spectrum = 0
        self.spectral_radius = 0
        self.second_largest_eigenvalue = 0
        self.energy = 0
        self.number_of_eigenvalues = 0
        self.neighbourhood_impurity = 0
        self.edge_impurity = 0
        self.label_entropy = 0
        self.identifier = identifier
        self.model = model
        self.nodes = {}
        self.edges = {}
        self.adjacency = {}

        self.node_type = node_type
        self.edge_type = edge_type
        self.subclass_type = literals.CLASS_NETWORK

        if fnss_topology is not None:
            assert node_type is not None, 'Node type is None'
            assert edge_type is not None, 'Edge type is None'
            self.calculate_attributes(fnss_topology)
            self.create_components(fnss_topology, node_type, edge_type)

    def create_components(self, fnss_topology, node_type, edge_type):
        """ Creates edge and node objects from fnss topology.

            Args:
                fnss_topology (fnss.Topology): Topology to create network from.
                node_type (class): Class of Nodes for network.
                edge_type (class): Class of Edges for network.
        """

        if 'model' in fnss_topology.graph:
            self.model = fnss_topology.graph['model']
        else:
            self.logger.warning(
                'Could not find model in fnss attributes. Make sure to set it yourself'
            )
        for node_id, attributes in fnss_topology.node.iteritems():
            self.nodes[node_id] = node_type(
                label=node_id,
                cpu=attributes['cpu'],
                latitude=attributes['latitude'] if 'latitude' in attributes else None,
                longitude=attributes['longitude'] if 'longitude' in attributes else None
            )
        count = 0
        for node_one, node_two, attr in fnss_topology.edges(data=True):
            if 'id' in attr:
                id = attr['id']
                count = id
            else:
                id = count
            count += 1
            self.edges[id] = edge_type(
                label=id,
                node_one_id=node_one,
                node_two_id=node_two,
                capacity=attr['capacity'],
                delay=attr['delay'] if 'delay' in attr else None,
                length=attr['length'] if 'length' in attr else None
            )
            if node_one in self.adjacency:
                self.adjacency[node_one].append(node_two)
            else:
                self.adjacency[node_one] = [node_two]
            if node_two in self.adjacency:
                self.adjacency[node_two].append(node_one)
            else:
                self.adjacency[node_two] = [node_one]

    def calculate_attributes(self, topology=None, bin_width=10):
        """
        Calculate various graph attributes.

        Note edge impurity, label entropy:
            Nodes of Graph are assumed to have `cpu` attribute being a floating
            point number. Edge impurity and label entropy are calculated based
            on this CPU value. Depending on how CPU values are generated (e.g.
            uniform distribution) and the size of the network, values will
            be sparse, thus entropy and impurity will be (close to) zero for
            all nodes/edges respectively. However CPU values might be nearly
            the same or very close (e.g. 70.1254 and 71.21).
            Therefore we are binning the CPU values.

        Args:
            topology (fnss.Topology): Graph to calculate attributes for. Must
                be connected.
            bin_width (float, optional): Bin CPU values of nodes to get a meaningful
                value for edge impurity and label entropy. Default is `10` assuming
                CPU values in `[0; 100)`. Set to `1` to disable binning.

        Returns:

        """
        if topology is None:
            topology = self.to_fnss_topology()

        degrees = topology.degree().values()
        if len(degrees) == 0:
            degrees = [0]
        average_neighbour_degree = np.mean(degrees)
        std_neighbour_degree = np.std(degrees)

        clustering_coefficients = nx.clustering(topology)
        if len(clustering_coefficients) == 0:
            clustering_coefficients = {'key': 0}
        average_clustering_coefficient = np.mean(clustering_coefficients.values())
        std_clustering_coefficient = np.std(clustering_coefficients.values())

        eccentricity = [max(nx.single_source_dijkstra_path_length(topology, i).values()) for i in topology.nodes_iter()]
        if len(eccentricity) == 0:
            eccentricity = [0]

        average_effective_eccentricity = np.mean(eccentricity)
        if np.isnan(average_effective_eccentricity):
            average_effective_eccentricity = 0
        std_effective_eccentricity = np.std(eccentricity)
        if np.isnan(std_effective_eccentricity):
            std_effective_eccentricity = 0
        max_effective_eccentricity = np.max(eccentricity)
        if np.isnan(max_effective_eccentricity):
            max_effective_eccentricity = 0
        min_effective_eccentricity = np.min(eccentricity)
        if np.isnan(min_effective_eccentricity):
            min_effective_eccentricity = 0

        closeness = nx.closeness_centrality(topology, distance='capacity').values()
        if len(closeness) == 0:
            closeness = [0]
        average_path_length = np.mean(closeness)
        std_path_length = np.std(closeness)
        percentage_central_points = np.mean(np.array(eccentricity) == max_effective_eccentricity)
        degrees = nx.degree(topology).values()
        if len(degrees) == 0:
            degrees = [0]
        percentage_end_points = np.mean(np.array(degrees) == 1)
        num_nodes = nx.number_of_nodes(topology)
        num_edges = nx.number_of_edges(topology)
        # Adjacency matrix is symmetric --> eigenvalues are real
        if len(topology.node) == 0:
            spectrum = 0
            spectral_radius = 0
            second_largest_eigenvalue = 0
            energy = 0
            number_of_eigenvalues = 0
        else:
            spectrum = nx.adjacency_spectrum(topology, weight='capacity').real
            spectral_radius = np.max(np.abs(spectrum))
            second_largest_eigenvalue = np.sort(spectrum)[1]
            energy = np.sum(np.power(spectrum, 2))
            number_of_eigenvalues = np.unique(np.array(spectrum)).size
        # Use CPU values for nodes as labels and calculate impurity and
        # label entropy from these values
        nodes = topology.node
        impurities = {}
        cpu_vals = {}

        for node in nodes.iterkeys():
            neighbourhood = topology[node].keys()
            label_home, rest = divmod(nodes[node]['cpu'], bin_width)

            if label_home not in cpu_vals:
                cpu_vals[label_home] = 1
            else:
                cpu_vals[label_home] += 1

            for neighbour in neighbourhood:
                # Exploit symmetry to save operations
                if neighbour < node:
                    continue
                else:
                    label_neighbor, rest = divmod(nodes[neighbour]['cpu'], bin_width)
                    if label_home != label_neighbor:
                        if node in impurities:
                            impurities[node] += 1.
                        else:
                            impurities[node] = 1.
                        if neighbour in impurities:
                            impurities[neighbour] += 1.
                        else:
                            impurities[neighbour] = 1.
        if len(impurities) == 0:
            impurities = np.array([0])
        else:
            impurities = impurities.values()
        neighbourhood_impurity = np.mean(impurities)
        # Divide by two because each edge is counted twice
        edge_impurity = 0 if num_edges == 0 else np.sum(impurities) / (2. * num_edges)
        tmp = np.array(cpu_vals.values())
        label_entropy = -np.sum(np.multiply(tmp, np.log2(tmp)))

        self.average_neighbour_degree =  average_neighbour_degree
        self.std_neighbour_degree =  std_neighbour_degree
        self.average_clustering_coefficient =  average_clustering_coefficient
        self.std_clustering_coefficient =  std_clustering_coefficient
        self.average_effective_eccentricity =  average_effective_eccentricity
        self.std_effective_eccentricity =  std_effective_eccentricity
        self.max_effective_eccentricity =  max_effective_eccentricity
        self.min_effective_eccentricity =  min_effective_eccentricity
        self.average_path_length =  average_path_length
        self.std_path_length =  std_path_length
        self.percentage_central_points =  percentage_central_points
        self.percentage_end_points =  percentage_end_points
        self.num_nodes =  num_nodes
        self.num_edges =  num_edges
        self.spectrum =  spectrum
        self.spectral_radius =  spectral_radius
        self.second_largest_eigenvalue =  second_largest_eigenvalue
        self.energy =  energy
        self.number_of_eigenvalues =  number_of_eigenvalues
        self.neighbourhood_impurity =  neighbourhood_impurity
        self.edge_impurity =  edge_impurity
        self.label_entropy =  label_entropy

    def todict(self):
        return {
            'average_neighbour_degree': self.average_neighbour_degree,
            'std_neighbour_degree': self.std_neighbour_degree,
            'average_clustering_coefficient': self.average_clustering_coefficient,
            'std_clustering_coefficient': self.std_clustering_coefficient,
            'average_effective_eccentricity': self.average_effective_eccentricity,
            'std_effective_eccentricity': self.std_effective_eccentricity,
            'max_effective_eccentricity': self.max_effective_eccentricity,
            'min_effective_eccentricity': self.min_effective_eccentricity,
            'average_path_length': self.average_path_length,
            'std_path_length': self.std_path_length,
            'percentage_central_points': self.percentage_central_points,
            'percentage_end_points': self.percentage_end_points,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'spectrum': self.spectrum,
            'spectral_radius': self.spectral_radius,
            'second_largest_eigenvalue': self.second_largest_eigenvalue,
            'energy': self.energy,
            'number_of_eigenvalues': self.number_of_eigenvalues,
            'neighbourhood_impurity': self.neighbourhood_impurity,
            'edge_impurity': self.edge_impurity,
            'label_entropy': self.label_entropy,
            'model': self.model,
            'subclass_type': self.subclass_type
        }

    def to_fnss_topology(self):
        """ Creates a FNSS topology from object.

            Returns:
                topo (fnss.Topology): Representation of network in FNSS.
        """
        topo = fnss.Topology()
        topo.graph = self.todict()

        for node in self.nodes.itervalues():
            topo.add_node(node.label, node.todict())
        for edge in self.edges.itervalues():
            topo.add_edge(edge.node_one_id, edge.node_two_id, edge.todict())
        return topo

    def copy(self):
        """ Creates a deep copy of object.

            Returns:
                network (Network): Copy of current network
        """
        return self.__class__(self.to_fnss_topology())

    def add_node(self, label, cpu, **kwargs):
        """ Adds node to newtork.

            Args:
                label (int): Node identifier.
                cpu (float): Cpu capacity of node.
                kwargs (dictionary): Additional arguments.
        """
        node =  self.node_type(
            label=label,
            cpu=cpu
        )
        for key, value in kwargs.iteritems():
            node.__setattr__(key, value)
        self.nodes[label] = node
        self.num_nodes += 1

    def add_edge(self, label, capacity, node_one_id, node_two_id, **kwargs):
        """ Adds edge to newtork.

            Args:
                id (int): Node identifier.
                capacity (float): Bandwidth capacity of edge.
                kwargs (dictionary): Additional arguments.
        """
        edge = self.edge_type(
            label=label,
            capacity=capacity,
            node_two_id=node_two_id,
            node_one_id=node_one_id,
            delay=None,
            length=None
        )
        for key, value in kwargs.iteritems():
            edge.__setattr__(key, value)
        self.edges[label] = edge
        self.num_edges += 1

    def save(self, **kwargs):
        """ Writes Object to the database.

            Raises:
                RuntimeError if implementor not set.
        """
        self.implementor.save_object(self, **kwargs)

    def get_edge(self, node_one, node_two):
        """ Searches for edge given nodes.

            Args:
                node_one (int): Identifier of first node.
                node_two (int): Identifier of second node.

            Returns:
                Edge if edge is found else None
        """
        nodes = [node_two, node_one]
        for edge in self.edges.itervalues():
            if (edge.node_one_id in nodes) and (edge.node_two_id in nodes):
                return edge
        return None


class VirtualNetwork(Network):

    def __init__(self, fnss_topology=None, implementor=None, identifier=None, model=None):
        """ Initializes object.

            Args:
                fnss_topology (fnss.Topology): Topology to create network from.
                identifier (object): Unique identifier.
                implementor (input_output.data_interface.vne.bridge.abstract_implementor.AbstractImplementor):
                    Derivation of `AbstractImplementor` handling input output
                    to data source.
                model (string): Model of network.
        """
        super(VirtualNetwork, self).__init__(
            fnss_topology=fnss_topology,
            node_type=VirtualNode,
            edge_type=VirtualEdge,
            implementor=implementor,
            identifier=identifier,
            model=model
        )
        self.subclass_type = literals.CLASS_VIRTUAL_NETWORK

    def calculate_attributes(self, topology=None, bin_width=10):
        """
        Calculate various graph attributes.

        Note edge impurity, label entropy:
            Nodes of Graph are assumed to have `cpu` attribute being a floating
            point number. Edge impurity and label entropy are calculated based
            on this CPU value. Depending on how CPU values are generated (e.g.
            uniform distribution) and the size of the network, values will
            be sparse, thus entropy and impurity will be (close to) zero for
            all nodes/edges respectively. However CPU values might be nearly
            the same or very close (e.g. 70.1254 and 71.21).
            Therefore we are binning the CPU values.

        Args:
            topology (fnss.Topology): Graph to calculate attributes for. Must
                be connected.
            bin_width (float, optional): Bin CPU values of nodes to get a meaningful
                value for edge impurity and label entropy. Default is `10` assuming
                CPU values in `[0; 100)`. Set to `1` to disable binning.

        Returns:

        """
        # TODO make this more flexible, so that one can configure it from a
        # file
        if topology is None:
            topology = self.to_fnss_topology()

        degrees = topology.degree().values()
        if len(degrees) == 0:
            degrees = [0]
        average_neighbour_degree = np.mean(degrees)

        eccentricity = [max(nx.single_source_dijkstra_path_length(topology, i).values()) for i in topology.nodes_iter()]
        if len(eccentricity) == 0:
            eccentricity = [0]
        max_effective_eccentricity = np.max(eccentricity)
        if np.isnan(max_effective_eccentricity):
            max_effective_eccentricity = 0

        closeness = nx.closeness_centrality(topology, distance='capacity').values()
        if len(closeness) == 0:
            closeness = [0]
        average_path_length = np.mean(closeness)

        num_nodes = nx.number_of_nodes(topology)
        num_edges = nx.number_of_edges(topology)
        # Adjacency matrix is symmetric --> eigenvalues are real
        if len(topology.node) == 0:
            spectral_radius = 0
            number_of_eigenvalues = 0
        else:
            spectrum = nx.adjacency_spectrum(topology, weight='capacity').real
            spectral_radius = np.max(np.abs(spectrum))
            number_of_eigenvalues = np.unique(np.array(spectrum)).size

        self.average_neighbour_degree =  average_neighbour_degree
        self.max_effective_eccentricity =  max_effective_eccentricity
        self.average_path_length =  average_path_length
        self.num_nodes =  num_nodes
        self.num_edges =  num_edges
        self.spectral_radius =  spectral_radius
        self.number_of_eigenvalues =  number_of_eigenvalues

    @property
    def requested_cpu(self):
        """ Total amount of CPU requested in network.

            Returns:
                cpu (float): total amount of CPU.
        """
        cpu = 0.
        for node in self.nodes.itervalues():
            cpu += node.cpu
        return cpu

    @property
    def requested_capacity(self):
        """ Total amount of capacity requested in network.

            Returns:
                capacity (float): total amount of requetsed capacity.
        """
        capacity = 0.
        for edge in self.edges.itervalues():
            capacity += edge.capacity
        return capacity

    @property
    def allocated_capacity(self):
        """ Returns total amount of allocated capacity for network.

            Returns:
                capacity (float): Total amount of allocated capacity.
        """
        capacity = 0.
        for edge in self.edges.itervalues():
            capacity += edge.allocated_capacity
        return capacity

    @property
    def allocated_cpu(self):
        """ Returns total amount of allocated CPU.

            Returns:
                cpu (float): total amount of allocated CPU.
        """
        cpu = 0.
        for node in self.nodes.itervalues():
            cpu += node.cpu
        return cpu

    def todict(self):
        """ Returns dictionary of attributes.

            Returns:
                dictionary
        """
        attributes = super(VirtualNetwork, self).todict()
        attributes['requested_cpu'] = self.requested_cpu,
        attributes['requested_capacity'] = self.requested_capacity,
#        attributes['allocated_cpu'] = self.allocated_cpu,
#        attributes['allocated_capacity'] = self.allocated_capacity
        return attributes


class PhysicalNetwork(Network):

    def __init__(self, fnss_topology=None, implementor=None, identifier=None, model=None):
        """ Initializes object.

            Args:
                fnss_topology (fnss.Topology): Topology to create network from.
                identifier (object): Unique identifier.
                implementor (input_output.data_interface.vne.bridge.abstract_implementor.AbstractImplementor):
                    Derivation of `AbstractImplementor` handling input output
                    to data source.
                model (string): Model of network.
        """
        super(PhysicalNetwork, self).__init__(
            fnss_topology=fnss_topology,
            node_type=PhysicalNode,
            edge_type=PhysicalEdge,
            implementor=implementor,
            identifier=identifier,
            model=model
        )
        self.subclass_type = literals.CLASS_PHYSICAL_NETWORK

    @property
    def free_cpu(self):
        """ Returns total amount of free CPU in network.

            Returns:
                cpu (float): Free CPU
        """
        cpu = 0.
        for node in self.nodes.itervalues():
            cpu += node.free_cpu
        return cpu

    @property
    def occupied_cpu(self):
        """ Returns total amount of occupied CPU in network.

            Returns:
                cpu (float): Occupied CPU.
        """
        cpu = 0.
        for node in self.nodes.itervalues():
            cpu += node.occupied_cpu
        return cpu

    @property
    def total_cpu(self):
        """ Returns total amount of CPU in Network.

            Returns:
                cpu (float): Total CPU capacity.
        """
        cpu = 0.
        for node in self.nodes.itervalues():
            cpu += node.cpu
        return cpu

    @property
    def free_capacity(self):
        """ Returns total mount of free capacity of network.

            Returns:
                capacity (float): Free capacity.
        """
        capacity = 0.
        for edge in self.edges.itervalues():
            capacity += edge.free_capacity
        return capacity

    @property
    def occupied_capacity(self):
        """ Returns total amount of occupied capacity in network.

            Returns:
                capacity (float): Occupied Capacity.
        """
        capacity = 0.
        for edge in self.edges.itervalues():
            capacity += edge.occupied_capacity
        return capacity

    @property
    def total_capacity(self):
        """ Returns total amount of capacity in network.

            Returns:
                capacity (float): Total amount of capacity.
        """
        capacity = 0.
        for edge in self.edges.itervalues():
            capacity += edge.capacity
        return capacity

    @property
    def total_mapped_nodes(self):
        """ Returns total number of mapped nodes.

            Returns:
                mapped (int): Number of mapped noes.
        """
        mapped = 0.
        for node in self.nodes.itervalues():
            mapped += node.total_mapped
        return mapped

    @property
    def currently_mapped_nodes(self):
        """ Returns total number of currently mapped nodes.

            Returns:
                mapped (int): Number of currently mapped nodes.
        """
        mapped = 0.
        for node in self.nodes.itervalues():
            mapped += node.currently_mapped
        return mapped

    @property
    def currently_mapped_edges(self):
        """ Returns total number of currently mapped edges.

            Returns:
                mapped (int): Number of currently mapped edges.
        """
        mapped = 0.
        for edge in self.edges.itervalues():
            mapped += edge.currently_mapped
        return mapped

    @property
    def total_mapped_edges(self):
        """ Returns number of in total mapped edges.

            Returns:
                mapped (int): Number of in total mapped edges.
        """
        mapped = 0.
        for edge in self.edges.itervalues():
            mapped += edge.total_mapped
        return mapped

    def impose_embedding(self, embedding):
        """ Applies changes of embedding to the substrate. Sets capacities and
            cpu values and other attributes affected.

            Args:
                embedding (scenario.simulation.vne.simulation_objects.Embedding):
                    Embedding object to apply to graph.
        """
        for node_embedding in embedding.node_embeddings:
            node = self.nodes[node_embedding.physical_node]
            node.currently_mapped += 1
            node.total_mapped += 1
            node.occupied_cpu += node_embedding.cpu
            node.free_cpu -= node_embedding.cpu

        for edge_embedding in embedding.edge_embeddings:
            # TODO: incidence matrix
            for n1, n2 in edge_embedding.physical_edges:
                edge = self.get_edge(n1, n2)
                edge.free_capacity -= edge_embedding.capacity
                edge.occupied_capacity += edge_embedding.capacity
                edge.currently_mapped += 1
                edge.total_mapped += 1

    def remove_embedding(self, embedding):
        """ Removes changes of embedding previously imposed on substrate. Sets
            capacities and cpu values and other attributes affected.

            Args:
                embedding (scenario.simulation.vne.simulation_objects.Embedding):
                    Embedding object to apply to graph.
        """
        for node_embedding in embedding.node_embeddings:
            # Actually this should work by manipulating the CPU of the node
            # embedding physical node object. But you never know...
            node = self.nodes[node_embedding.physical_node]
            node.currently_mapped -= 1
            node.occupied_cpu -= node_embedding.cpu
            node.free_cpu += node_embedding.cpu

        for edge_embedding in embedding.edge_embeddings:
            for n1, n2 in edge_embedding.physical_edges:
                edge = self.get_edge(n1, n2)
                edge.free_capacity += edge_embedding.capacity
                edge.occupied_capacity -= edge_embedding.capacity
                edge.currently_mapped -= 1

    def todict(self):
        """ Returns attributes as dictionary.

            Returns:
                dictionary
        """
        attributes = super(PhysicalNetwork, self).todict()
        attributes['free_cpu'] = self.free_cpu,
        attributes['occupied_cpu'] = self.occupied_cpu,
        attributes['total_cpu'] = self.total_cpu,
        attributes['free_capacity'] = self.free_capacity,
        attributes['occupied_capacity'] = self.occupied_capacity,
        attributes['total_capacity'] = self.total_capacity,
        attributes['total_mapped_nodes'] = self.total_mapped_nodes,
        attributes['total_mapped_edges'] = self.total_mapped_edges,
        attributes['currently_mapped_nodes'] = self.currently_mapped_nodes,
        attributes['currently_mapped_edges'] = self.currently_mapped_edges
        return attributes


