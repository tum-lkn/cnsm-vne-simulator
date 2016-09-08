import literals
import time
import algorithms.abstract_algorithm
import numpy as np
import errors.homebrewederrors as custom_errors


class RnnFilter(algorithms.abstract_algorithm.AbstractAlgorithm):
    """ Performs filtering before applying the actual embedding algorithm.
    """

    def __init__(self, type, serialized_model, aux_params, successor_name,
                 comment=None, physical_network=None, **kwargs):
        """ Initializes object.

        Args:
            type (string): Type of learning model. Momentarily supported: breze_rnn
            serialized_model (string): Path to pickled model.
            aux_params (dict): Dictionary with additional parameter for the
                learning model.
            successor_name (string): Name of algorithm to be used after filtering
                step is performed.
            comment (string, optional): A comment to the model.
            physical_network (network_model.vne.networkmodel.PhysicalNetwork):
                Substrate network.
            **kwargs: Additional parameter for successor algorithm.

        Returns:
            None

        """
        import algorithms.factory as algo_factory
        import scenario.simulation.simulation_objects as simulation_objects
        super(RnnFilter, self).__init__(physical_network=physical_network)
        self.successor = algo_factory.AlgorithmFactory.produce(
            name=successor_name,
            substrate=physical_network,
            **kwargs
        )
        self.serialized_model = serialized_model
        self.aux_params = aux_params
        self.type = type
        self.learning_model = simulation_objects.BrezeSupervisedRnnModel(
            modelpath=self.serialized_model,
            **self.aux_params
        )
        self.rnn_prediction_time = None
        self.substrate_feature_calculation_time = None
        self.vnr_featur_calculation_time = None

    def _filter_request(self, current_vnr, simulation):
        start = time.clock()
        current_vnr.calculate_attributes()
        self.vnr_feature_calculation_time = time.clock() - start

        start = time.clock()
        simulation.current_state.inverse_network.calculate_attributes()
        self.substrate_feature_calculation_time = time.clock() - start

        total_cap = simulation.current_state.inverse_network.total_capacity
        total_cpu = simulation.current_state.inverse_network.total_cpu
        X = np.array([
            simulation.current_state.inverse_network.currently_mapped_edges,
            simulation.current_state.inverse_network.currently_mapped_nodes,
            float(simulation.current_state.inverse_network.free_capacity) / total_cap,
            float(simulation.current_state.inverse_network.free_cpu) / total_cpu,
            float(simulation.current_state.inverse_network.occupied_capacity) / total_cap,
            float(simulation.current_state.inverse_network.occupied_cpu) / total_cpu,
            total_cap,
            total_cpu,
            simulation.current_state.inverse_network.average_neighbour_degree,
            simulation.current_state.inverse_network.average_clustering_coefficient,
            simulation.current_state.inverse_network.average_effective_eccentricity,
            simulation.current_state.inverse_network.max_effective_eccentricity,
            simulation.current_state.inverse_network.min_effective_eccentricity,
            simulation.current_state.inverse_network.average_path_length,
            simulation.current_state.inverse_network.percentage_central_points,
            simulation.current_state.inverse_network.percentage_end_points,
            simulation.current_state.inverse_network.num_nodes,
            simulation.current_state.inverse_network.num_edges,
            simulation.current_state.inverse_network.spectral_radius,
            simulation.current_state.inverse_network.second_largest_eigenvalue,
            simulation.current_state.inverse_network.number_of_eigenvalues,
            simulation.current_state.inverse_network.neighbourhood_impurity,
            simulation.current_state.inverse_network.edge_impurity,
            simulation.current_state.inverse_network.label_entropy,
            simulation.current_state.inverse_network.std_neighbour_degree,
            simulation.current_state.inverse_network.std_clustering_coefficient,
            simulation.current_state.inverse_network.std_effective_eccentricity,
            simulation.current_state.inverse_network.std_path_length,
            simulation.current_state.inverse_network.energy,
            current_vnr.num_nodes,
            current_vnr.spectral_radius,
            current_vnr.max_effective_eccentricity,
            current_vnr.average_neighbour_degree,
            current_vnr.number_of_eigenvalues,
            current_vnr.average_path_length,
            current_vnr.num_edges
        ])
        start = time.clock()
        prediction = self.learning_model.predict(X).flatten()
        self.rnn_prediction_time = time.clock() - start

        if prediction[0] < literals.VNR_ACCEPTANCE_THRESHOLD:
            return True
        else:
            return False

    def run(self, simulation, virtual_network, **kwargs):
        """ Execute Run.

        Args:
            simulation (scenario.simulation.concrete_simulations.VneSimulation):
                Simulation algorithm is applied in.
            virtual_network (networkmodel.vne.network_model.VirtualNetwork):
                VNR to be embedded
            **kwargs: arguments for successor.

        Returns:

        """
        from scenario.simulation.simulation_objects import Embedding
        if simulation.simulation_steps == 1:
            # If this is the first simulation step, simply accept
            embedding = self.successor.run(virtual_network=virtual_network, **kwargs)
            self.substrate_feature_calculation_time = 0
            self.rnn_prediction_time = 0
            self.vnr_feature_calculation_time = 0
        elif self._filter_request(current_vnr=virtual_network, simulation=simulation):
            embedding = self.successor.run(virtual_network=virtual_network, **kwargs)
        else:
            embedding = Embedding()
            embedding.run_time = 0.
            embedding.setup_time = 0.
            embedding.solving_time = 0.
            embedding.vnr_classification = 3

        embedding.substrate_feature_extraction_time = self.substrate_feature_calculation_time
        embedding.rnn_prediction_time = self.rnn_prediction_time
        embedding.vnr_feature_extraction_time = self.vnr_feature_calculation_time
        embedding.run_time += self.vnr_feature_calculation_time \
                              + self.substrate_feature_calculation_time \
                              + self.rnn_prediction_time
        return embedding
