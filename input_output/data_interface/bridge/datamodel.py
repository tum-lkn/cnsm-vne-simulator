import peewee
import logging

logging.basicConfig(level=logging.INFO)
peewee_logger = logging.getLogger('peewee')
peewee_logger.setLevel(logging.INFO)
peewee_logger.addHandler(logging.StreamHandler())

db = peewee.MySQLDatabase(None)

config = {
    'db_name': None,
    'host': None,
    'port': None,
    'user': None,
    'password': None
}


def set_connection_parameter(database, host, port, user, passwd):
    config['database'] = database
    config['host'] = host
    config['port'] = port
    config['user'] = user
    config['password']= passwd
    db.init(
        database=database,
        host=host,
        port=port,
        user=user,
        passwd=passwd
    )


def connect():
    db.connect()


def disconnect():
    db.close()


class ConnectionManager(object):

    def __init__(self, input_config):
        """ Initializes object.

        Args:
            input_config (input_output.data_interface.interface_config.PeeweeInterfaceConfig):
                configuration object for database

        Returns:
            None

        """
        self.database = input_config.database
        self.host = input_config.host
        self.port = input_config.port
        self.user = input_config.user
        self.passwd = input_config.passwd
        self.config_object = input_config
        self._config = input_config.as_dict()

    @property
    def config(self):
        return self._config.copy()

    @property
    def db(self):
        """ return database

        Returns: peewee.MySQLDatabase

        """
        return db

    def as_dict(self):
        return self.config

    def connect(self):
        if (config['db_name'] != self.database) or (config['host'] != self.host) or \
                (config['user'] != self.user) or (config['port'] != self.port) or \
                (config['password'] != self.passwd):
            set_connection_parameter(
                database=self.database,
                host=self.host,
                port=self.port,
                passwd=self.passwd,
                user=self.user
            )
            connect()

    def disconnect(self):
        disconnect()


class BaseModel(peewee.Model):
    class Meta:
        database = db


class Experiment(BaseModel):
    experiment_id = peewee.PrimaryKeyField()
    description = peewee.CharField()

    class Meta:
        db_table = 'Experiment'


class NetworkGenerationSetting(BaseModel):
    network_generation_setting_id = peewee.PrimaryKeyField()
    model = peewee.TextField(null=False)
    connectivity = peewee.DecimalField(default=None, null=True)
    m = peewee.IntegerField(default=None, null=True)
    m0 = peewee.IntegerField(default=None, null=True)
    alpha = peewee.DecimalField(default=None, null=True)
    beta = peewee.DecimalField(default=None, null=True)
    minx = peewee.DecimalField(default=None, null=True)
    maxx = peewee.DecimalField(default=None, null=True)
    miny = peewee.DecimalField(default=None, null=True)
    maxy = peewee.DecimalField(default=None, null=True)
    order = peewee.IntegerField(default=None, null=True)
    min_order = peewee.IntegerField(default=None, null=True)
    max_order = peewee.IntegerField(default=None, null=True)
    min_capacity = peewee.DecimalField(null=False)
    max_capacity = peewee.DecimalField(null=False)
    capacity_generation = peewee.TextField(null=False)
    min_cpu = peewee.DecimalField(null=False)
    max_cpu = peewee.DecimalField(null=False)
    min_edge_distance = peewee.DecimalField(default=None, null=True)
    max_edge_distance = peewee.DecimalField(default=None, null=True)
    delay_per_km = peewee.DecimalField(default=None, null=True)
    is_substrate = peewee.BooleanField(null=False)

    class Meta:
        db_table = 'NetworkGenerationSetting'


class AlgorithmSetting(BaseModel):
    algorithm_setting_id = peewee.PrimaryKeyField()
    name = peewee.TextField(null=False)
    parameter = peewee.TextField(null=False)

    class Meta:
        db_table = 'AlgorithmSetting'


class ProcessSetting(BaseModel):
    process_setting_id = peewee.PrimaryKeyField()
    arrival_rate = peewee.IntegerField(null=False)
    distribution = peewee.CharField(null=False, default='negative_exponential')
    num_requests = peewee.IntegerField(null=True)
    type = peewee.CharField(null=False)

    class Meta:
        db_table = 'ProcessSetting'


class EventGenerationSetting(BaseModel):
    event_generation_setting_id = peewee.PrimaryKeyField()
    #arrival_process_setting = peewee.ForeignKeyField(
    #    rel_model=ProcessSetting,
    #    to_field='process_setting_id',
    #    related_name='arrival_process'
    #)
    #service_process_setting = peewee.ForeignKeyField(
    #    rel_model=ProcessSetting,
    #    to_field='process_setting_id',
    #    related_name='service_process'
    #)
    network_setting = peewee.ForeignKeyField(
        rel_model=NetworkGenerationSetting,
        to_field='network_generation_setting_id'
    )

    class Meta:
        db_table = 'EventGenerationSetting'


class ProcessToEventSetting(BaseModel):
    process_to_event_setting = peewee.PrimaryKeyField()
    process_setting = peewee.ForeignKeyField(
        rel_model=ProcessSetting,
        to_field='process_setting_id'
    )
    event_generation_setting = peewee.ForeignKeyField(
        rel_model=EventGenerationSetting,
        to_field='event_generation_setting_id'
    )

    class Meta:
       db_table = 'ProcessToEventSetting'


class Network(BaseModel):
    network_id = peewee.PrimaryKeyField()
    fnss_attributes = peewee.TextField(null=True)
    model = peewee.CharField(null=False)
    subclass_type=peewee.IntegerField(null=False)
    average_neighbour_degree = peewee.IntegerField(null=False)
    average_clustering_coefficient = peewee.FloatField(null=False)
    average_effective_eccentricity = peewee.FloatField(null=False)
    max_effective_eccentricity = peewee.FloatField(null=False)
    min_effective_eccentricity = peewee.FloatField(null=False)
    average_path_length = peewee.FloatField(null=False)
    percentage_central_points = peewee.FloatField(null=False)
    percentage_end_points = peewee.FloatField(null=False)
    num_nodes = peewee.IntegerField(null=False)
    num_edges = peewee.IntegerField(null=False)
    spectral_radius = peewee.FloatField(null=False)
    second_largest_eigenvalue = peewee.FloatField(null=False)
    energy = peewee.FloatField(null=False)
    number_of_eigenvalues = peewee.IntegerField(null=False)
    neighbourhood_impurity = peewee.FloatField(null=False)
    edge_impurity = peewee.FloatField(null=False)
    label_entropy = peewee.FloatField(null=False)
    std_neighbour_degree = peewee.FloatField(null=False)
    std_clustering_coefficient = peewee.FloatField(null=False)
    std_effective_eccentricity = peewee.FloatField(null=False)
    std_path_length = peewee.FloatField(null=False)

    class Meta:
        db_table = 'Network'


class NetworkGenerationToNetwork(BaseModel):
    network_generation_to_network = peewee.PrimaryKeyField()
    network_generation_setting = peewee.ForeignKeyField(
        rel_model=NetworkGenerationSetting,
        to_field='network_generation_setting_id'
    )
    network = peewee.ForeignKeyField(
        rel_model=Network,
        to_field='network_id'
    )

    class Meta:
        db_table = 'NetworkGenerationToNetwork'


class Node(BaseModel):
    node_id = peewee.PrimaryKeyField()
    network = peewee.ForeignKeyField(
        Network,
        related_name='nodes',
        on_delete='CASCADE',
        to_field='network_id'
    )
    label = peewee.IntegerField()
    attributes = peewee.CharField()

    class Meta:
        db_table = 'Node'


class Edge(BaseModel):
    edge_id = peewee.PrimaryKeyField()
    label = peewee.IntegerField()
    network = peewee.ForeignKeyField(
        Network,
        related_name='edges',
        on_delete='CASCADE',
        to_field='network_id'
    )
    node_one = peewee.IntegerField()
    node_two = peewee.IntegerField()
    attributes = peewee.CharField()

    class Meta:
        db_table = 'Edge'


class EventHeap(BaseModel):
    event_heap_id = peewee.PrimaryKeyField()
    event_generation_setting = peewee.ForeignKeyField(
        rel_model=EventGenerationSetting,
        to_field='event_generation_setting_id'
    )

    class Meta:
        db_table = 'EventHeap'


class Event(BaseModel):
    event_id = peewee.PrimaryKeyField()
    event_heap = peewee.ForeignKeyField(
        rel_model=EventHeap,
        to_field='event_heap_id',
        related_name='events',
        on_delete='CASCADE'
    )
    network = peewee.ForeignKeyField(
        Network,
        to_field='network_id',
        on_delete='CASCADE'
    )
    time = peewee.FloatField()
    lifetime = peewee.FloatField()

    class Meta:
        db_table = 'Event'


class LearningModel(BaseModel):
    learning_model_id = peewee.PrimaryKeyField()
    type = peewee.TextField()
    serialized_model = peewee.TextField()
    aux_params = peewee.TextField()
    comment = peewee.TextField()

    class Meta:
        db_table = 'LearningModel'


class Scenario(BaseModel):
    scenario_id = peewee.PrimaryKeyField()
    algorithm_setting = peewee.ForeignKeyField(
            rel_model=AlgorithmSetting,
            to_field='algorithm_setting_id'
    )
    network_generation_setting = peewee.ForeignKeyField(
            rel_model=NetworkGenerationSetting,
            to_field='network_generation_setting_id',
            help_text='Generation Settings for substrate'
    )
    event_generation_setting = peewee.ForeignKeyField(
            rel_model=EventGenerationSetting,
            to_field='event_generation_setting_id'
    )
    learning_model = peewee.ForeignKeyField(
            rel_model=LearningModel,
            to_field='learning_model_id',
            null=True
    )
    #experiment = peewee.ForeignKeyField(
    #        rel_model=Experiment,
    #        to_field='experiment_id'
    #)
    description = peewee.CharField(null=True)

    class Meta:
        db_table = 'Scenario'


class RunConfiguration(BaseModel):
    run_configuration_id = peewee.PrimaryKeyField()
    scenario = peewee.ForeignKeyField(
            rel_model=Scenario,
            to_field='scenario_id'
    )
    network = peewee.ForeignKeyField(
        Network,
        related_name='network',
        to_field='network_id',
        help_text='Foreign key to actual network to use as substrate'
    )
    event_heap = peewee.ForeignKeyField(
        EventHeap,
        to_field='event_heap_id'
    )

    class Meta:
        db_table = 'RunConfiguration'


class RunExecution(BaseModel):
    run_execution_id = peewee.PrimaryKeyField()
    run_configuration = peewee.ForeignKeyField(
        rel_model=RunConfiguration,
        to_field='run_configuration_id'
    )
    stage_of_execution = peewee.IntegerField(default=0, null=False)
    num_successful_embeddings = peewee.IntegerField(default=0, null=False)
    num_failed_embeddings = peewee.IntegerField(default=0, null=False)
    num_infeasible_embeddings = peewee.IntegerField(default=0, null=False)
    execution_time = peewee.FloatField(default=0., null=False)
    num_cores = peewee.IntegerField(default=0, null=False)
    priority = peewee.IntegerField(default=10, null=False)
    num_filtered_embeddings = peewee.IntegerField(default=0, null=False)

    class Meta:
        db_table = 'RunExecution'


class Embedding(BaseModel):
    # Add new values here
    embedding_id = peewee.PrimaryKeyField()
    vnr_classification = peewee.IntegerField()
    optimality_gap = peewee.DoubleField()
    run_time = peewee.FloatField()
    setup_time = peewee.FloatField(default=0)
    solving_time = peewee.FloatField(default=0)
    requested_cpu = peewee.FloatField()
    allocated_cpu = peewee.FloatField()
    requested_nodes = peewee.IntegerField()
    requested_capacity = peewee.FloatField()
    allocated_capacity = peewee.FloatField()
    embedded_edges = peewee.IntegerField()
    requested_edges = peewee.IntegerField()
    substrate_feature_extraction_time = peewee.FloatField(null=True)
    rnn_prediction_time = peewee.FloatField(null=True),
    vnr_feature_extraction_time = peewee.FloatField(null=True)
    network = peewee.ForeignKeyField(
        Network,
        related_name='embedded_network',
        null=True
    )

    fallback_used = peewee.BooleanField(null=True)
    time_hf_exe = peewee.FloatField(null=True)
    time_hf_prep = peewee.FloatField(null=True)
    time_nodemap = peewee.FloatField(null=True)
    time_edgemap = peewee.FloatField(null=True)
    time_nodemap_original = peewee.FloatField(null=True)
    time_edgemap_original = peewee.FloatField(null=True)
    hf_number_selected = peewee.IntegerField(null=True)
    hf_num_iterations = peewee.IntegerField(null=True)

    class Meta:
        db_table = 'Embedding'


class NodeEmbedding(BaseModel):
    node_embedding_id = peewee.PrimaryKeyField()
    physical_node = peewee.IntegerField()
    virtual_node = peewee.IntegerField()
    cpu = peewee.FloatField()
    embedding = peewee.ForeignKeyField(
        Embedding,
        related_name='node_embeddings',
        on_delete='CASCADE',
        to_field='embedding_id'
    )

    class Meta:
        db_table = 'NodeEmbedding'


class EdgeEmbedding(BaseModel):
    edge_embedding_id = peewee.PrimaryKeyField()
    virtual_node_one = peewee.IntegerField()
    virtual_node_two = peewee.IntegerField()
    capacity = peewee.FloatField()
    embedding = peewee.ForeignKeyField(
        Embedding,
        related_name='edge_embeddings',
        on_delete='CASCADE',
        to_field='embedding_id'
    )

    class Meta:
        db_table = 'EdgeEmbedding'


class EdgeEmbeddingPart(BaseModel):
    edge_embedding_part_id = peewee.PrimaryKeyField()
    edge_embedding = peewee.ForeignKeyField(
        EdgeEmbedding,
        related_name='edge_embedding_parts',
        on_delete='CASCADE',
        to_field='edge_embedding_id'
    )
    physical_node_one = peewee.IntegerField()
    physical_node_two = peewee.IntegerField()

    class Meta:
        db_table = 'EdgeEmbeddingPart'


class EventOccurrence(BaseModel):
    event_occurrence_id = peewee.PrimaryKeyField()
    time = peewee.FloatField()
    subclass_type = peewee.IntegerField(null=False)
    occurred = peewee.BooleanField(default=False, null=False)
    event_id = peewee.IntegerField(default=None, null=True)
    embedding = peewee.ForeignKeyField(
        Embedding,
        related_name='event_occurrences',
        null=True,
        on_delete='CASCADE'
    )
    run_execution = peewee.ForeignKeyField(
        RunExecution,
        related_name='occurrences',
        to_field='run_execution_id',
        on_delete='CASCADE'
    )

    class Meta:
        db_table = 'EventOccurrence'


class SubstrateState(BaseModel):
    substrate_state_id = peewee.PrimaryKeyField()
    event_occurrence = peewee.ForeignKeyField(
        EventOccurrence,
        related_name='after_state',
        to_field='event_occurrence_id',
        on_delete='CASCADE'
    )
    network = peewee.ForeignKeyField(
        rel_model=Network,
        to_field='network_id',
        on_delete='CASCADE'
    )
    currently_mapped_edges = peewee.IntegerField()
    currently_mapped_nodes = peewee.IntegerField()
    free_capacity = peewee.FloatField()
    free_cpu = peewee.FloatField()
    occupied_capacity = peewee.FloatField()
    occupied_cpu = peewee.FloatField()
    total_capacity = peewee.FloatField()
    total_cpu = peewee.FloatField()

    class Meta:
        db_table = 'SubstrateState'


class NodeState(BaseModel):
    node_state_id = peewee.PrimaryKeyField()
    substrate_state = peewee.ForeignKeyField(
        SubstrateState,
        related_name='node_states',
        on_delete='CASCADE',
        to_field='substrate_state_id'
    )
    free_cpu = peewee.FloatField()
    total_cpu = peewee.FloatField()
    total_mapped = peewee.IntegerField(default=0)
    currently_mapped = peewee.IntegerField(default=0)
    label = peewee.IntegerField()

    class Meta:
        db_table = 'NodeState'


class EdgeState(BaseModel):
    edge_state_id = peewee.PrimaryKeyField()
    substrate_state = peewee.ForeignKeyField(
        SubstrateState,
        related_name='edge_states',
        on_delete='CASCADE',
        to_field='substrate_state_id'
    )
    free_capacity = peewee.FloatField()
    total_capacity = peewee.FloatField()
    total_mapped = peewee.IntegerField(default=0)
    currently_mapped = peewee.IntegerField(default=0)
    node_one = peewee.IntegerField()
    node_two = peewee.IntegerField()

    class Meta:
        db_table = 'EdgeState'

