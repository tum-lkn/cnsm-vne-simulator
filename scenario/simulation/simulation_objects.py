"""
Implements objects used during simulation such as Embedding, Substrate State, etc.
"""
import numpy as np
import network_model.networkmodel as networkmodel
import literals
import logging
import heapq
import cPickle as pkl
import algorithms.factory as vne_algorithms
logging.basicConfig(level=logging.DEBUG)

__author__ = 'Patrick Kalmbach'


class SimulationObject(object):

    @classmethod
    def from_database(cls, implementor, object_id):
        """ Retrieves object from database given unique identifier

            Args:
                implementor (input_output.vne.bridge.abstract_implementor.AbstractImplementor):
                    Some object specifying sink. For more details refer to class
                    ObjectFactory in peewee_implementors.py
                object_id (object): Some unique identifier
        """
        object = implementor.get_object(object_id)
        object.implementor = implementor
        return object

    @classmethod
    def get_identifier(cls, implementor, **kwargs):
        """ Retrieves identifier of object given attributes.

            Args:
                implementor (input_output.vne.bridge.abstract_implementor.AbstractImplementor):
                    Some object specifying sink. For more details refer to class
                    ObjectFactory in peewee_implementors.py
        """
        return implementor.get_object_id(**kwargs)

    @classmethod
    def exists(cls, implementor, **kwargs):
        """ Checks whether object exists in database.

            Args:
                implementor (input_output.vne.bridge.abstract_implementor.AbstractImplementor):
                    Some object specifying sink. For more details refer to class
                    ObjectFactory in peewee_implementors.py
        """
        return implementor.exists_object(**kwargs)

    def __init__(self, identifier=None, implementor=None):
        """ Initializes object.

            Args:
                identifier (object): Identifier of some type.
                implementor (input_output.data_interface.vne.bridge.abstract_implementor.AbstractImplementor):
                    Implementor for object.
        """
        self.identifier = identifier
        self._implementor = implementor
        name = str(self.__class__)
        name = name[name.rfind('.') + 1]
        self.logger = logging.getLogger(name)

    def save(self, **kwargs):
        """ Writes Object to the database.

            Args:
                kwargs: Additional attributes
        """
        self.implementor.save_object(self, **kwargs)

    def update(self, **kwargs):
        """ Updates object in datasource.

        Args:
            **kwargs: Additional arguments.

        Returns:
            None
        """
        self.implementor.update_object(self, **kwargs)

    @property
    def implementor(self):
        """

        Returns:
            input_output.data_interface.vne.bridge.abstract_implementor.AbstractImplementor

        """
        return self._implementor

    @implementor.setter
    def implementor(self, implementor):
        """

        Args:
            implementor (input_output.data_interface.vne.bridge.abstract_implementor.AbstractImplementor):
                Concrete implementor
        """
        self._implementor = implementor


class NodeEmbedding(object):

    def __init__(self, physical_node, virtual_node, cpu_demand, identifier=None):
        """ Initializes object.

            Args:
                physical_node (int): label of Physical node.
                virtual_node (int): Label of Virtual node
                cpu_demand (float): CPU Demand of virtual node
        """
        self.identifier = identifier
        self.physical_node = physical_node
        self.virtual_node = virtual_node
        self.cpu = cpu_demand

    def todict(self):
        return {
            'physical_node': self.physical_node,
            'virtual_node': self.virtual_node,
            'cpu': self.cpu
        }


class EdgeEmbedding(object):

    def __init__(self, physical_edges, virtual_edge, capacity_demand, identifier=None):
        """ Initializes object.

            Args:
                physical_edges (list of tuples): Labels of nodes embedded virtual
                    edge is implemented on.
                virtual_edge (tuple): Labels of adjacent virtual nodes.
                capacity_demand (float): Capacity demand of virtual edge.
        """
        self.identifier = identifier
        self.physical_edges = physical_edges
        self.virtual_edge = virtual_edge
        self.capacity = capacity_demand

    def todict(self):
        return {
            'virtual_node_one': self.virtual_edge[0],
            'virtual_node_two': self.virtual_edge[1],
            'capacity': self.capacity
        }


class Embedding(SimulationObject):

    def __init__(self, vnr_classification=None,
                 event=None, identifier=None, implementor=None, **kwargs):
        """ Initializes object.

            vnr (network_model.vne.networkmodel.VirtualNetwork): Virtual Network
                topology representing VNR.
            substrate (network_model.vne.networkmodel.PhysicalNetwork): Substrate
                topology.
            vnr_classification (int): Result of attempting to map VNR. Must be in
                `{0, 1, 2, 3}` representing _no feasible solution in given time_,
                _accepted_, _infeasible_, _filtered out_ respectively.
            event (TODO type of this):
            kwargs: Additional parameter.
        """
        super(Embedding, self).__init__(identifier, implementor)

        self.exclude = [
            'embedded_topology',
            'event',
            'node_embeddings',
            'edge_embeddings'
        ]

        self.node_embeddings = []
        self.edge_embeddings = []
        self.embedded_topology = networkmodel.Network(model='embedded_topology')
        self.embedded_topology.subclass_type = literals.CLASS_EMBEDDED_NETWORK

        self.requested_cpu = 0
        self.allocated_cpu = 0
        self.requested_capacity = 0
        self.allocated_capacity = 0
        self.requested_edges = 0
        self.requested_nodes = 0
        self.embedded_edges = 0

        if vnr_classification is not None:
            self.vnr_classification = vnr_classification
        else:
            self._vnr_classification = None
        self.event = event
        self.optimality_gap = kwargs['optimality_gap'] \
            if 'optimality_gap' in kwargs else 0
        self.run_time = kwargs['run_time'] if 'run_time' in kwargs else 0
        self.setup_time = kwargs['setup_time'] if 'setup_time' in kwargs else 0
        self.solving_time = kwargs['solving_time'] if 'solving_time' in kwargs \
            else 0
        self.substrate_feature_extraction_time = kwargs['substrate_feature_extration_time'] \
            if 'substrate_feature_extraction_time' in kwargs else 0
        self.rnn_prediction_time = kwargs['rnn_prediction_time'] if \
            'rnn_prediction_time' in kwargs else 0
        self.vnr_feature_extraction_time = kwargs['vnr_feature_extraction_time'] \
            if 'vnr_feature_extraction_time' in kwargs else 0

    @property
    def vnr_classification(self):
        """ Returns vnr_classification indicating whether or not request has
            been accepted, rejected or filtered.

            Returns:
                vnr_classification (int): Integer in {0, 1, 2, 3}
        """

        return self._vnr_classification

    @vnr_classification.setter
    def vnr_classification(self, vnr_classification):
        """ Sets value of vnr_classification attribute.

            Args:
                vnr_classification (int): Result of attempting to map VNR. Must be in
                    `{0, 1, 2, 3}` representing _no feasible solution in given time_,
                    _accepted_, _infeasible_, _filtered out_ respectively.
        """
        assert vnr_classification in [
            literals.VNR_ACCEPTED,
            literals.VNR_FILTERED,
            literals.VNR_INFEASIBLE,
            literals.VNR_NO_SOLUTION
        ], 'vnr_classification must be in {0, 1, 2, 3}'
        self._vnr_classification = vnr_classification

    def create_embedding_objects(self, vnr, substrate):
        """ Generates Node and Edge Embedding objects as well as embedded
            network in substrate.

            Args:
                vnr (network_model.vne.networkmodel.VirtualNetwork): Representation
                    of virtual network request.
                substrate (network_model.vne.networkmodel.PhysicalNetwork):
                    Representation of physical network request.
        """
        #self.embedded_topology = networkmodel.Network()
        #for node in vnr.nodes.itervalues():
        #    self.node_embeddings.append(NodeEmbedding(
        #        physical_node=substrate.nodes[node.mapped_to],
        #        virtual_node=node,
        #        cpu_demand=node.cpu
        #    ))
        #    self.embedded_topology.add_node(
        #        label=node.label,
        #        cpu=node.cpu
        #    )
        #for edge in vnr.edges.itervalues():
        #    carriers = []
        #    for n1, n2 in edge.mapped_to:
        #        if n1 not in self.embedded_topology.nodes:
        #            self.embedded_topology.add_node(n1, cpu=0.)
        #        if n2 not in self.embedded_topology.nodes:
        #            self.embedded_topology.add_node(n2, cpu=0.)
        #        self.embedded_topology.add_edge(
        #            label=self.embedded_topology.num_edges,
        #            node_one_id=n1,
        #            node_two_id=n2,
        #            capacity=edge.capacity
        #        )
        #        self.embedded_edges += 1
        #        carriers.append(substrate.get_edge(n1, n2))
        #    self.edge_embeddings.append(EdgeEmbedding(
        #        virtual_edge=edge,
        #        physical_edges=carriers,
        #        capacity_demand=edge.capacity
        #    ))
        raise NotImplementedError

    def add_node_embedding(self, physical_node_label, virtual_node_label, cpu_demand):
        """ Add a node embedding.

            Args:
                physical_node_label (int): Node, virtual node is mapped to.
                virtual_node_label (int): Node of an vnr that is being mapped.
                cpu_demand (float): CPU demanded by virtual node
        """
        if self.embedded_topology is None:
            self.embedded_topology = networkmodel.Network()
            self.embedded_topology.subclass_type = literals.CLASS_EMBEDDED_NETWORK

        self.node_embeddings.append(NodeEmbedding(
            physical_node=physical_node_label,
            virtual_node=virtual_node_label,
            cpu_demand=cpu_demand
        ))
        self.requested_cpu += cpu_demand
        self.allocated_cpu += cpu_demand
        self.requested_nodes += 1

        self.embedded_topology.add_node(
            label=physical_node_label,
            cpu=cpu_demand
        )

    def add_edge_embedding(self, physical_edges, virtual_edge, capacity_demand):
        """ Adds an embedded path to the embedding.

            Args:
                physical_edges (list of tuples): Labels of nodes embedded virtual
                    edge is implemented on.
                virtual_edge (tuple): Labels of adjacent virtual nodes.
        """
        self.requested_edges += 1
        self.embedded_edges += len(physical_edges)
        self.requested_capacity += capacity_demand
        self.allocated_capacity += capacity_demand * len(physical_edges)

        self.edge_embeddings.append(EdgeEmbedding(
            virtual_edge=virtual_edge,
            physical_edges=physical_edges,
            capacity_demand=capacity_demand
        ))

        for node_one_id, node_two_id in physical_edges:
            if node_one_id not in self.embedded_topology.nodes:
                self.embedded_topology.add_node(node_one_id, 0.0)
            if node_two_id not in self.embedded_topology.nodes:
                self.embedded_topology.add_node(node_two_id, 0.0)

            self.embedded_topology.add_edge(
                label=self.embedded_topology.num_edges + 1,
                node_one_id=node_one_id,
                node_two_id=node_two_id,
                capacity=capacity_demand
            )

    def todict(self):
        """ Creates dictionary of attributes.

            Returns:
                dictionary
        """
        dictionary = {}
        for key, value in self.__dict__.iteritems():
            if key not in self.exclude:
                if key.startswith('_'):
                    key = key[1:]
                dictionary[key] = value
            else:
                continue
        return dictionary


class SubstrateState(SimulationObject):

    def __init__(self, substrate, implementor=None, identifier=None):
        """ Initializes Object.

            Args:
                substrate (network_model,vne.PhysicalNetwork): Representation
                    of substrate.
        """
        super(SubstrateState, self).__init__(identifier, implementor)
        self.substrate = substrate
        self.inverse_network = networkmodel.PhysicalNetwork(model='inverse_network')
        self.inverse_network.subclass_type = literals.CLASS_INVERSE_NETWORK
        self.currently_mapped_edges = 0
        self.currently_mapped_nodes = 0
        self.free_capacity = 0
        self.occupied_capacity = 0
        self.total_capacity = 0
        self.free_cpu = 0
        self.occupied_cpu = 0
        self.total_cpu = 0

        for node in substrate.nodes.itervalues():
            self.currently_mapped_nodes += node.currently_mapped
            self.free_cpu += node.free_cpu
            self.total_cpu += node.cpu
            self.occupied_cpu += node.occupied_cpu
            if node.currently_mapped > 0:
                self.inverse_network.add_node(**node.todict())

        for edge in substrate.edges.itervalues():
            self.currently_mapped_edges += edge.currently_mapped
            self.free_capacity += edge.free_capacity
            self.occupied_capacity += edge.occupied_capacity
            self.total_capacity += edge.capacity

            if edge.currently_mapped > 0:
                if edge.node_one_id not in self.inverse_network.nodes:
                    self.inverse_network.add_node(edge.node_one_id, cpu=0.)
                if edge.node_two_id not in self.inverse_network.nodes:
                    self.inverse_network.add_node(edge.node_two_id, cpu=0.)
                self.inverse_network.add_edge(**edge.todict())
        self.inverse_network.calculate_attributes()

    def todict(self):
       return {
           'currently_mapped_edges': self.currently_mapped_edges,
           'currently_mapped_nodes': self.currently_mapped_nodes,
           'free_capacity': self.free_capacity,
           'occupied_capacity': self.occupied_capacity,
           'total_capacity': self.total_capacity,
           'free_cpu': self.free_cpu,
           'occupied_cpu': self.occupied_cpu,
           'total_cpu': self.total_cpu
       }


class EventQueue(SimulationObject):
    def __init__(self, implementor=None, identifier=None):
        super(EventQueue, self).__init__(identifier, implementor)
        self.heap = []

    def heappush(self, simulation_event):
        heapq.heappush(
            self.heap,
            (
                simulation_event.arrival_time,
                simulation_event.priority,
                simulation_event
            )
        )

    def heappop(self):
        return heapq.heappop(self.heap)[2]

    def __iter__(self):
        """
        Returns: scenario.simulation.vne.simulation_objects.SimulationEvent

        """
        while len(self.heap) > 0:
            yield self.heappop()

    def save(self, **kwargs):
        """ Writes Object to the database.

            Args:
                event_generation_object (EventGenerationSettings): Object
                    representing the generation settings.

            Raises:
                AssertionError if event_generation_object not set.
        """
        assert 'event_generation_object' in kwargs, 'event generation expected for saving event' \
                                         'queue'
        event_generation = kwargs['event_generation_object']
        self.implementor.save_object(
            self,
            event_generation=event_generation
        )
        for event in self:
            event.save(event_queue=self)


class SimulationEvent(SimulationObject):
    def __init__(self, arrival_time, identifier=None, implementor=None):
        super(SimulationEvent, self).__init__(identifier, implementor)
        self.arrival_time = arrival_time
        self.priority = 0

    def handle(self, simulation):
        """ Attempt to handle event.

            Args:
                simulation (scenario.simulation.concrete_simulations.VneSimulation):
                    Simulation Event is part of.

            Returns:
                event_occurrence (scenario.simulation.vne.simulation_objects.EventOccurrence):
                    Occurrence Object for event.
        """
        raise NotImplementedError


class ArrivalEvent(SimulationEvent):
    def __init__(self, arrival_time, lifetime, virtual_network, identifier=None,
                 implementor=None):
        """ Initializes object.

            Args:
                arrival_time (float): Time Event is going to arrive/happen.
                lifetime (float): Time Event effects stay in system.
                virtual_network (network_model.vne.networkmodel.VirtualNetwork):
                    Network arriving
        """
        super(ArrivalEvent, self).__init__(arrival_time, identifier, implementor)
        self.lifetime = lifetime
        self.virtual_network = virtual_network
        self.subclass_type = literals.CLASS_ARRIVAL_EVENT

    def save(self, **kwargs):
        self.virtual_network.save()
        self.implementor.save_object(self, **kwargs)

    def handle(self, simulation):
        """ Attempt to handle event.

            Args:
                simulation (scenario.simulation.concrete_simulations.VneSimulation):
                    Simulation Event is part of.

            Returns:
                event_occurrence (scenario.simulation.vne.simulation_objects.EventOccurrence):
                    Occurrence Object for event.
        """
        simulation.algorithm.substrate = simulation.substrate
        embedding = simulation.algorithm.run(
            simulation=simulation,
            virtual_network=self.virtual_network,
            num_cores=simulation.num_cores
        )
        simulation.current_embedding = embedding

        if embedding.vnr_classification == literals.VNR_ACCEPTED:
            simulation.substrate.impose_embedding(
                embedding=embedding
            )
            simulation.run_execution.num_successful_embeddings += 1
            simulation.event_queue.heappush(
                DepartureEvent(
                    arrival_time=self.arrival_time + self.lifetime,
                    embedding=embedding
                )
            )
        elif embedding.vnr_classification == literals.VNR_INFEASIBLE:
            simulation.run_execution.num_infeasible_embeddings += 1
        elif embedding.vnr_classification == literals.VNR_NO_SOLUTION:
            simulation.run_execution.num_failed_embeddings += 1
        elif embedding.vnr_classification == literals.VNR_FILTERED:
            simulation.run_execution.num_filtered_embeddings += 1
        else:
            self.logger.warning('Unknown vnr_classification {}'.format(embedding.vnr_classification))

        occurrence = EventOccurrence(
            simulation_event=self,
            run_execution=simulation.run_execution,
            embedding=embedding,
            time=self.arrival_time
        )
        return occurrence


class DepartureEvent(SimulationEvent):

    def __init__(self, arrival_time, embedding, identifier=None, implementor=None):
        super(DepartureEvent, self).__init__(arrival_time, identifier, implementor)
        self.embedding = embedding
        self.subclass_type = literals.CLASS_DEPARTURE_EVENT

    def handle(self, simulation):
        """ Attempt to handle event.

            Args:
                simulation (scenario.simulation.concrete_simulations.VneSimulation):
                    Simulation Event is part of.

            Returns:
                event_occurrence (scenario.simulation.vne.simulation_objects.EventOccurrence):
                    Occurrence Object for event.
        """
        simulation.substrate.remove_embedding(
            embedding=self.embedding
        )
        simulation.current_embedding = None
        return EventOccurrence(
            simulation_event=self,
            run_execution=simulation.run_execution,
            embedding=self.embedding,
            time=self.arrival_time
        )


class EventOccurrence(SimulationObject):

    def __init__(self, simulation_event, run_execution, embedding, time=None,
                 identifier=None, implementor=None):
        """ Initializes object.
        """
        super(EventOccurrence, self).__init__(identifier, implementor)
        self.subclass_type = None
        self.network = None
        self.time = time
        if type(simulation_event) == ArrivalEvent:
            self.subclass_type = literals.CLASS_ARRIVAL_EVENT
            self.network = simulation_event.virtual_network
            self.event_id = simulation_event.identifier
        elif type(simulation_event) == DepartureEvent:
            self.subclass_type = literals.CLASS_DEPARTURE_EVENT
            self.event_id = None
        else:
            raise RuntimeError('Unkown simulation event class {}'.format(str(type(simulation_event))))
        self.run_execution = run_execution
        self.embedding = embedding


class LearningModel(SimulationObject):
    def __init__(self, identifier=None, implementor=None):
        super(LearningModel, self).__init__(identifier, implementor)

    def predict(self, X):
        raise NotImplementedError('Not implemented')


class BrezeSupervisedRnnModel(LearningModel):
    def __init__(self, mean, std, din, hlayers, dout, hidden_transfer,
                 output_transfer, loss, optimizer, modelpath, identifier=None,
                 implementor=None):
        import breze.learn.rnn as rnn
        super(BrezeSupervisedRnnModel, self).__init__(identifier, implementor)
        self.mean = mean
        self.std = std
        self.model = rnn.SupervisedRnn(
            din,
            hlayers,
            dout,
            hidden_transfer,
            output_transfer,
            loss=loss,
            optimizer=optimizer,
            batch_size=50,
            pooling='mean'
        )
        with open(modelpath, 'rb') as fh:
            self.model.parameters.data[...] = pkl.load(fh)

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, X.shape[0])
        if len(X.shape) == 2:
            X = np.divide(np.subtract(X, self.mean), self.std)
            X = X.reshape(1, X.shape[0], X.shape[1])
        else:
            raise ValueError('Array with more than 2 dimensions passed')
        Y = self.model.predict(X)
        return Y


class Algorithm(SimulationObject):

    def __init__(self, name, parameter, identifier=None, implementor=None):
        """ Initializes object.

            Args:
                name (string): Name of the algorithm.
                parameter (dict): Additinaly parameter. Content of this dictinary
                    will be made available as object attributes.
                identifier (object, optional): Some unique identifier.
        """
        super(Algorithm, self).__init__(identifier, implementor)
        self.name = name
        self.parameter = parameter
        self.algorithm = vne_algorithms.AlgorithmFactory.produce(
            name=self.name,
            substrate=None,
            **self.parameter
        )
        self._substrate = None
        for key, value in parameter.iteritems():
            # Create attributes from parameter
            setattr(self, key, value)

    @property
    def substrate(self):
        return self._substrate

    @substrate.setter
    def substrate(self, substrate):
        """ Substrate Network.

            Args:
                substrate (network_model.vne.networkmodel.PhysicalNetwork)
        """
        self._substrate = substrate
        self.algorithm.physical = substrate

    def todict(self):
        return self.parameter.copy()

    def run(self, simulation, virtual_network, num_cores):
        """ Execute the algorithm and return an embedding.

            Args:
                simulation (scenario.simulation.concrete_simulations.VneSimulation):
                    Simulation algorithm is applied in.
                virtual_network (network_model.vne.networkmodel.VirtualNetwork):
                    vnr to embed.
                num_cores (int): Number of cores (for gurobi) to run on.
            Returns:
                embedding (Embedding)
        """
        embedding = self.algorithm.run(
            simulation=simulation,
            virtual_network=virtual_network,
            num_cores=num_cores
        )
        return embedding


class StochasticProcess(SimulationObject):

    @classmethod
    def from_type(cls, distribution, type, arrival_rate=None, num_requests=None,
                 identifier=None, implementor=None):
        if type == literals.PROCESS_TYPE_ARRIVAL:
            return ArrivalProcess(distribution, arrival_rate, num_requests,
                                  identifier, implementor)
        elif type == literals.PROCESS_TYPE_SERVICE:
            return ServiceProcess(distribution, arrival_rate, num_requests,
                                  identifier, implementor)

    def __init__(self, distribution, arrival_rate=None, num_requests=None,
                 identifier=None, implementor=None):
        """ Initializes Object.

            Args:
                distribution (string): Type of distribution, momentarily only
                    negative_exponential supported.
                arrival_rate (double, optional): Must be set when used in conjunction
                    with negative_exponential. Describes average time between
                    to events.
                num_requests (int): Size of queue, used for arrival process.
        """
        super(StochasticProcess, self).__init__(identifier, implementor)
        self.distribution = distribution
        self.arrival_rate = arrival_rate
        self.num_requests = num_requests
        self.type = None

    def todict(self):
        return {
            'distribution': self.distribution,
            'arrival_rate': self.arrival_rate,
            'num_requests': self.num_requests,
            'type': self.type
        }


class ArrivalProcess(StochasticProcess):

    def __init__(self, distribution, arrival_rate=None, num_requests=None,
                 identifier=None, implementor=None):
        """ Initializes Object.

            Args:
                distribution (string): Type of distribution, momentarily only
                    negative_exponential supported.
                arrival_rate (double, optional): Must be set when used in conjunction
                    with negative_exponential. Describes average time between
                    to events.
                num_requests (int): Size of queue, used for arrival process.
                identifier (object): Unique identifier.
                implementor (input_output.data_interface.vne.bridge.abstract_implementor.AbstractImplementor):
                    implementor handling data I/O.
        """
        super(ArrivalProcess, self).__init__(distribution, arrival_rate, num_requests,
                                             identifier, implementor)
        self.type = literals.PROCESS_TYPE_ARRIVAL


class ServiceProcess(StochasticProcess):

    def __init__(self, distribution, arrival_rate=None, num_requests=None,
                 identifier=None, implementor=None):
        """ Initializes Object.

            Args:
                distribution (string): Type of distribution, momentarily only
                    negative_exponential supported.
                arrival_rate (double, optional): Must be set when used in conjunction
                    with negative_exponential. Describes average time between
                    to events.
                num_requests (int): Size of queue, used for arrival process.
                identifier (object): Unique identifier.
                implementor (input_output.data_interface.vne.bridge.abstract_implementor.AbstractImplementor):
                    implementor handling data I/O.
        """
        super(ServiceProcess, self).__init__(distribution, arrival_rate, num_requests,
                                             identifier, implementor)
        self.type = literals.PROCESS_TYPE_SERVICE


class NetworkGenerationSettings(SimulationObject):

    def __init__(self,  model, connectivity, min_capacity, max_capacity,
                 capacity_generation, min_cpu, max_cpu, is_substrate,
                 m=None, m0=None, alpha=None, beta=None, minx=None, maxx=None,
                 miny=None, maxy=None, order=None, min_order=None,
                 max_order=None, min_edge_distance=None, max_edge_distance=None,
                 delay_per_km=None, identifier=None, implementor=None):
        super(NetworkGenerationSettings, self).__init__(identifier, implementor)
        self.model = model
        self.connectivity = connectivity
        self.m = m
        self.m0 = m0
        self.alpha = alpha
        self.beta = beta
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.order = order
        self.min_order = min_order
        self.max_order = max_order
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.capacity_generation = capacity_generation
        self.min_cpu = min_cpu
        self.max_cpu = max_cpu
        self.min_edge_distance = min_edge_distance
        self.max_edge_distance = max_edge_distance
        self.delay_per_km = float(delay_per_km)
        self.is_substrate = is_substrate

    def todict(self):
        return {
            'model': self.model,
            'connectivity': self.connectivity,
            'm': self.m,
            'm0': self.m0,
            'alpha': self.alpha,
            'beta': self.beta,
            'minx': self.minx,
            'maxx': self.maxx,
            'miny': self.miny,
            'maxy': self.maxy,
            'order': self.order,
            'min_order': self.min_order,
            'max_order': self.max_order,
            'min_capacity': self.min_capacity,
            'max_capacity': self.max_capacity,
            'capacity_generation': self.capacity_generation,
            'min_cpu': self.min_cpu,
            'max_cpu': self.max_cpu,
            'min_edge_distance': self.min_edge_distance,
            'max_edge_distance': self.max_edge_distance,
            'delay_per_km': self.delay_per_km,
            'is_substrate': self.is_substrate
        }


class EventGenerationSettings(SimulationObject):

    def __init__(self, arrival_process, service_process,
                 network_generation_settings, identifier=None, implementor=None):
        """ Initializes object.

            Args:
                arrival_process (StochasticProcess): Process generating inter
                    arrival times of events.
                service_process (StochasticProcess): Process generating life
                    time, i.e. service time, of events.
                network_generation_settings (NetworkGenerationSettings):
                    Settings describing how networks are created.
                identifier (object): Some unique identifier.
        """
        super(EventGenerationSettings, self).__init__(identifier, implementor)
        self.arrival_process = arrival_process
        self.service_process = service_process
        self.network_generation_settings = network_generation_settings


class Scenario(SimulationObject):

    def __init__(self, algorithm_setting, learning_model_id, network_generation,
                 event_generation, identifier=None, implementor=None):
        super(Scenario, self).__init__(identifier, implementor)
        self.algorithm_setting = algorithm_setting
        self.learning_model_id = learning_model_id
        self.event_generation = event_generation
        self.network_generation = network_generation


class RunConfiguration(SimulationObject):

    def __init__(self, scenario, network, event_queue, identifier=None, implementor=None):
        super(RunConfiguration, self).__init__(identifier, implementor)
        self.scenario = scenario
        self.network = network
        self.event_queue = event_queue


class RunExecution(SimulationObject):

    def __init__(self, identifier=None, implementor=None):
        """ Initializes object.
        """
        super(RunExecution, self).__init__(identifier, implementor)
        self.stage_of_execution = 0
        self.num_successful_embeddings = 0
        self.num_failed_embeddings = 0
        self.num_infeasible_embeddings = 0
        self.execution_time = 0.
        self.num_cores = None
        self.priority = 10
        self.num_filtered_embeddings = 0

    def todict(self):
        """ Return dictionary of objects attributes.

            Returns:
                dict
        """
        return {
            'stage_of_execution' : self.stage_of_execution,
            'num_successful_embeddings' : self.num_successful_embeddings,
            'num_failed_embeddings' : self.num_failed_embeddings,
            'num_infeasible_embeddings' : self.num_infeasible_embeddings,
            'execution_time' : self.execution_time,
            'num_cores' : self.num_cores,
            'priority' : self.priority,
            'num_filtered_embeddings' : self.num_filtered_embeddings
        }

    def save(self, **kwargs):
        """ Write object to output interface.

            Args:
                run_configuration_id (object): Identifier of respective run
                    configuration.

                Raises:
                    AssertionError if run_configuration_id not present.
        """
        assert 'run_configuration_id' in kwargs, 'run_configuration_id required' \
                                                 'to save RunExecution object'
        super(RunExecution, self).save(**kwargs)


class SimulationStepResult(object):
    """ Container class, for each step in the simulation (each event occurrence)
        hold the possible objects that might need storing away.
    """

    def __init__(self, embedding=None, event_occurrence=None,
                 substrate_state=None):
        self.embedding = embedding
        self.event_occurrence = event_occurrence
        self.substrate_state = substrate_state

