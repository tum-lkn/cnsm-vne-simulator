import vnet_tum_wrapper
import literals


class AbstractAlgorithm(object):
    """ Base class for vne algorithms. Makes available general methods and functions.

        Attributes:
            _physical (vnet_tum_wrapper.VnettumWrapper): Representation of
                substrate networkmodel object as a (wrapped) fnss topology.
                Necessary for backward compatibility.
            virtual (vnet_tum_wrapper.VnettumWrapper): Representation of VNR.
                Passed to and set in the run method each time it is called.

    """

    def __init__(self, physical_network):
        """ Initializes object.

        Args:
            physical_network (network_model.vne.networkmodel.PhysicalNetwork):
                Networkobject.

        Returns:
            None

        """
        if physical_network is None:
            self._physical = None
        else:
            self.physical = physical_network
        self.virtual = None

    @property
    def physical(self):
        return self._physical

    @physical.setter
    def physical(self, physical_network):
        """ Sets the substrate network.

        Args:
            physical_network (network_model.vne.networkmodel.PhysicalNetwork):
                Networkobject.

        Returns:
            None

        """
        self._physical = vnet_tum_wrapper.VnettumWrapper(
            network=physical_network.to_fnss_topology(),
            substrate=True
        )
        if hasattr(self, 'pGraph'):
            # two stage embedding and subsequenc classes
            setattr(
                self,
                'pGraph',
                self._physical.getGraph()
            )
        elif hasattr(self, 'successor'):
            # Rnn Filter
            self.successor.physical = physical_network

    def create_embedding_object(self, vnr_classification, nodes, edges, **kwargs):
        """
        Create an embedding object that stores all information about an embedding
        Args:
            vnr_classification (int): Whether vnr has been accepted, was
                infeasible or filtered or no solution existed.
            nodes: Dict contating the node mapping
            edges: List contating the edge mapping
            *kwargs: Additional data that will be written to the object

        Returns: An Embedding object that contains all entered data

        """
        from scenario.simulation.simulation_objects import Embedding
        emb = Embedding()

        emb.vnr_classification = vnr_classification

        # actual node and link mapping
        if vnr_classification == literals.VNR_ACCEPTED:
            for virtual_node in nodes:
                emb.add_node_embedding(
                    physical_node_label=nodes[virtual_node],
                    virtual_node_label=virtual_node,
                    cpu_demand=self.virtual.getCPU(virtual_node)
                )
            for (virtual_edge, physical_edges) in edges.iteritems():
                emb.add_edge_embedding(
                    physical_edges=physical_edges,
                    virtual_edge=virtual_edge,
                    capacity_demand=self.virtual.getBandwidth(
                        virtual_edge[0],
                        virtual_edge[1]
                    )
                )

        # other information
        for arg in kwargs:
            emb.__setattr__(arg, kwargs[arg])

        return emb

