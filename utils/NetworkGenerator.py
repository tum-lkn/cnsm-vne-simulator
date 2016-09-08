""" Contains a network generator creating networks for VNE scenarios.
"""
import fnss
import numpy as np
import networkx as nx
import logging
import literals


class NetworkGenerator(object):
    """
    Generate synthetic (random) networks. The following models are available:
    """

    @classmethod
    def method_factory(cls, method):
        """ Returns a function creating a network according to specified
            model.

            Args:
                method (string): Network model to use. Must be in
                    {ER, STAR, FM, BA, WAX}

            Returns:
                function: Function returning a fnss.Topology object.
        """
        if method == 'erdoes_renyi' or method == 'ER':
            return NetworkGenerator().erdoes_renyi
        elif method == 'star' or method == 'STAR':
            return NetworkGenerator().star
        elif method == 'full_mesh' or method == 'FM':
            return NetworkGenerator().full_mesh
        elif method == 'barabasi_albert' or method == 'BA':
            return NetworkGenerator().barabasi_albert
        elif method == 'waxman2' or method == 'WAX':
            return NetworkGenerator().waxman2
        else:
            raise KeyError('Unknown network model {}'.format(method))

    @classmethod
    def remove_unnecessary_attributes(cls, args):
        #necessary = ['vnr', 'min_bw', 'max_bw', 'bandwidth', 'min_cpu',
        #             'max_cpu', 'min_distance', 'max_distance', 'delay',
        #             'substrate', 'minx', 'maxx', 'miny', 'maxy']
        #keys = args.keys()
        #for k in keys:
        #    if k not in necessary:
        #        args.pop(k)
        pass

    def __init__(self):
        """
        Initializes object.
        """
        self.logger = logging.getLogger(str(self.__class__))
        self.logger.setLevel(logging.DEBUG)

    def generate_attributes(self, vnr, min_capacity, max_capacity, capacity_generation,
                            min_cpu, max_cpu, min_edge_distance=None, max_edge_distance=None,
                            delay_per_km=0.005, is_substrate=False, minx=None, maxx=None,
                            miny=None, maxy=None, **kwargs):
        """
        Sets the following edge attributes:
            * capacity [Mb/s]
            * delay [ms]
            * length [km]
            * mapping - zero if substrate else empty list
            * free_capacity if substrate is True
        Delay is calculated based on attribute delay and length of edge.

        Sets the following node attributes:
            * cpu
            * mapping
            * free_cpu if substrate is True
        If the network was created using the Waxman2 method, a longitude and
        latitude attribute will be present for the nodes

        Args:
            vnr (Vnr): Object of type Vnr.
            min_bw (int): Minimal value for bandwidth on edge.
            max_bw (int): Maximal value for bandwidth on edge.
            bandwidth (string): {uniform, power} - How bandwidth should be
                generated.
                if uniform is chosen distribution follows uniform distribution,
                if power is chosen distribution follows a power law.
            min_cpu (int): Minimal value for CPU capacity.
            max_cpu (int): Maximal value for CPU capacity.
            min_distance (int): Minimal length of an edge.
            max_distance (int): Maximal length of an edge.
            delay (float, optional): Delay per kilometer of cable length
            substrate (optional, bool): Whether it is a substrate network or not

        """
        if delay_per_km is None:
            delay_per_km = 0.05
        if 'distance_unit' not in vnr.graph:
            vnr.graph['distance_unit'] = 'km'
        bws = range(int(min_capacity), int(max_capacity) + 1)
        if capacity_generation == 'uniform':
            fnss.set_capacities_random_uniform(
                topology=vnr,
                capacities=bws
            )
        elif capacity_generation == 'power':
            fnss.set_capacities_random_power_law(
                topology=vnr,
                capacities=bws
            )
        for attributes in vnr.node.itervalues():
            attributes['cpu'] = int(np.random.uniform(min_cpu, max_cpu))
            attributes['mapping'] = 0
            if is_substrate:
                attributes['free_cpu'] = attributes['cpu']
                attributes['total_mapped'] = 0
                attributes['currently_mapped'] = 0
                if ('latitude' not in attributes) and (minx is not None) and (maxx is not None):
                    attributes['latitude'] = np.random.uniform(minx, maxx)
                if ('longitude' not in attributes) and (minx is not None) and (maxx is not None):
                    attributes['longitude'] = np.random.uniform(miny, maxy)

        for i, j, attr in vnr.edges(data=True):
            if 'length' not in attr:
                if 'longitude' in vnr.node[i] and 'latitude' in vnr.node[i]:
                    attr['length'] = np.sqrt(
                        np.square(vnr.node[i]['latitude'] - vnr.node[j]['latitude']) +
                        np.square(vnr.node[i]['longitude'] - vnr.node[j]['longitude'])
                    )
                else:
                    if (min_edge_distance is None) and (max_edge_distance is None):
                        pass
                    else:
                        attr['length'] = np.random.uniform(min_edge_distance, max_edge_distance)
            attr['mapping'] = 0 if is_substrate else []
            if is_substrate:
                attr['free_capacity'] = attr['capacity']
                attr['total_mapped'] = 0
                attr['currently_mapped'] = 0
        fnss.set_delays_geo_distance(vnr, specific_delay=delay_per_km)
        fnss.set_weights_delays(vnr)

    def erdoes_renyi(self, **args):
        """
        Creates a random Erdoes-Renyi Graph.

        Args:
            order (int): Number of nodes.
            connectivity (float): Probability that two nodes are connected.
                Must be in (0; 1].
            args (dict): Attributes for property generation:
                min_bw (int): Minimal value for bandwidth on edge.
                max_bw (int): Maximal value for bandwidth on edge.
                bandwidth (string): {uniform, power} - How bandwidth should be
                    generated. If uniform is chosen distribution follows uniform
                    distribution, if power is chosen distribution follows a
                    power law.
                min_cpu (int): Minimal value for CPU capacity.
                max_cpu (int): Maximal value for CPU capacity.
                min_distance (int): Minimal length of an edge.
                max_distance (int): Maximal length of an edge.
                delay (float, optional): Delay per kilometer of cable length.
                substrate (optional, bool): Whether it is a substrate network.
        Returns: FNSS Graph

        """
        unconnected = True
        num_nodes = args.pop('order')
        connectivity = args.pop('connectivity')
        vnr = None
        while unconnected:
            vnr = fnss.erdos_renyi_topology(
                n=num_nodes,
                p=connectivity
            )
            unconnected = not nx.is_connected(vnr)
        self.remove_unnecessary_attributes(args)
        self.generate_attributes(vnr, **args)
        vnr.graph['model'] = literals.NETWORK_MODEL_ERDOES_RENYI
        return vnr

    def star(self, order, **args):
        """
        Creates a Star-Topology
        Args:
            order (int): Number of nodes.
            args (dict): Attributes for property generation:
                min_bw (int): Minimal value for bandwidth on edge.
                max_bw (int): Maximal value for bandwidth on edge.
                bandwidth (string): {uniform, power} - How bandwidth should be
                    generated. If uniform is chosen distribution follows uniform
                    distribution, if power is chosen distribution follows a
                    power law.
                min_cpu (int): Minimal value for CPU capacity.
                max_cpu (int): Maximal value for CPU capacity.
                min_distance (int): Minimal length of an edge.
                max_distance (int): Maximal length of an edge.
                delay (float, optional): Delay per kilometer of cable length.
                substrate (optional, bool): Whether it is a substrate network.

        Returns: FNSS object

        """
        vnr = fnss.star_topology(order - 1)
        self.remove_unnecessary_attributes(args)
        self.generate_attributes(vnr, **args)
        vnr.graph['model'] = literals.NETWORK_MODEL_STAR
        return vnr

    def full_mesh(self, order, **args):
        """
        Returns a fully connected graph.

        Args:
            order (int): Number of nodes.
            args (dict): Attributes for property generation:
                min_bw (int): Minimal value for bandwidth on edge.
                max_bw (int): Maximal value for bandwidth on edge.
                bandwidth (string): {uniform, power} - How bandwidth should be
                    generated. If uniform is chosen distribution follows uniform
                    distribution, if power is chosen distribution follows a
                    power law.
                min_cpu (int): Minimal value for CPU capacity.
                max_cpu (int): Maximal value for CPU capacity.
                min_distance (int): Minimal length of an edge.
                max_distance (int): Maximal length of an edge.
                delay (float, optional): Delay per kilometer of cable length.
                substrate (optional, bool): Whether it is a substrate network.

        Returns: FNSS object

        """
        vnr = fnss.full_mesh_topology(order)
        self.remove_unnecessary_attributes(args)
        self.generate_attributes(vnr, **args)
        vnr.graph['model'] = literals.NETWORK_MODEL_FULL_MESH
        return vnr

    def barabasi_albert(self, order, m=None, m0=None, **args):
        """
        Creates a scale free graph after the Barabasi-Albert model.

        Args:
            order (int): Number of nodes.
            m (int): Number of nodes a new node connects to.
            m0 (int): Number of initially connected nodes
            args (dict): Attributes for property generation:
                min_bw (int): Minimal value for bandwidth on edge.
                max_bw (int): Maximal value for bandwidth on edge.
                bandwidth (string): {uniform, power} - How bandwidth should be
                    generated. If uniform is chosen distribution follows uniform
                    distribution, if power is chosen distribution follows a
                    power law.
                min_cpu (int): Minimal value for CPU capacity.
                max_cpu (int): Maximal value for CPU capacity.
                min_distance (int): Minimal length of an edge.
                max_distance (int): Maximal length of an edge.
                delay (float, optional): Delay per kilometer of cable length.
                substrate (optional, bool): Whether it is a substrate network.

        Returns: FNSS object

        """
        unconnected = True
        if m is None or m0 is None:
            m0 = 2
            m = 1

            if order > 2:
                m0 = int(np.random.uniform(2, order - 1))
                m = int(np.random.uniform(1, m0 - 1))
        vnr = None
        while unconnected:
            vnr = fnss.barabasi_albert_topology(
                n=order,
                m=m,
                m0=m0
            )
            unconnected = not nx.is_connected(vnr)
        self.remove_unnecessary_attributes(args)
        self.generate_attributes(vnr, **args)
        vnr.graph['model'] = literals.NETWORK_MODEL_BARABASI_ALBERT
        return vnr

    def waxman2(self, order, alpha, beta, minx, miny, maxx, maxy, **kwargs):
        """
        Creates a random network following a power law distribution.

        Args:
            order (int): Number of nodes.
            alpha (float): Must be in (0,1]. Higher alpha increase difference
                between density of long and short links. Waxman found 0.3 to
                be a good value for this.
            beta (float): Must be in (0; 1]. Regulates link density, higher
                beta higher link density.
            minx (int): Minimal x-coordinate on grid.
            miny (int): Minimal y-coordinate on grid.
            maxx (int): Maximal x-coordinate on grid.
            maxy (int): Maximal y-coordinate on grid.
            args (dict): Attributes for property generation:
                min_bw (int): Minimal value for bandwidth on edge.
                max_bw (int): Maximal value for bandwidth on edge.
                bandwidth (string): {uniform, power} - How bandwidth should be
                    generated. If uniform is chosen distribution follows uniform
                    distribution, if power is chosen distribution follows a
                    power law.
                min_cpu (int): Minimal value for CPU capacity.
                max_cpu (int): Maximal value for CPU capacity.
                delay (float, optional): Delay per kilometer of cable length.
                substrate (optional, bool): Whether it is a substrate network.

        Returns: FNSS object

        """
        unconnected = True
        vnr = None
        while unconnected:
            vnr = fnss.waxman_2_topology(
                n=order,
                alpha=beta,  # This is confusing, in paper and literature
                beta=alpha,  # alpha and beta usually are reversed to FNSS terminology!
                domain=(
                    minx,
                    miny,
                    maxx,
                    maxy
                )
            )
            unconnected = not nx.is_connected(vnr)
        self.remove_unnecessary_attributes(kwargs)
        self.generate_attributes(vnr, **kwargs)
        vnr.graph['model'] = literals.NETWORK_MODEL_WAXMAN
        return vnr
