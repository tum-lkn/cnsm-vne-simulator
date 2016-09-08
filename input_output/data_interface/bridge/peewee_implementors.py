""" To be able to use different interface technologies (File, different DBMs)
    The bride pattern is used to differentiate implementation from abstraction.
"""
import errors.homebrewederrors as custom_exceptions
import input_output.data_interface.bridge.datamodel as datamodel
import literals
from input_output.data_interface.bridge.abstract_implementor import AbstractImplementor
import network_model.networkmodel as networkmodel
import scenario.simulation.simulation_objects as simulation_objects


class PeeweeImplementor(AbstractImplementor):

    def _assemble_restrictions(self, constraints, table):
        """ Given a dictionary of constrains, i.e. values for attributes,
            create a list of restrictions (logical expressions) for use in a
            where clause for peewee.

            Args:
                constraints (dict): Key-value pairs.
                table (input_output.data_interface.vne.bridge.datamodel.BaseModel):
                    Class/Relation to which constraints apply.

            Returns:
                list
        """
        restrictions = []
        for key, value in constraints.iteritems():
            if not hasattr(table, key):
                self.logger.warning('Unknown attribute {} for relation {}, spelling mistake?'.format(
                    key,
                    str(table)
                ))
                continue
            if type(value) in [str, unicode]:
                restrictions.append(getattr(table, key) % value)
            elif type(value) in [float, int, bool]:
                restrictions.append(getattr(table, key) == value)
            else:
                raise RuntimeError(('Undefined type {}, check operation at '
                                   'http://docs.peewee-orm.com/en/latest/peewee/querying.html '
                                   'and add respective type').format(str(type(value))))
        return restrictions


class NetworkImplementor(PeeweeImplementor):

    def __init__(self, db_connection):
        super(NetworkImplementor, self).__init__(connection=db_connection)

    def save_object(self, object, **kwargs):
        """ Stores object in database.

            Args:
                object (network_model.vne.networkmodel.Network):
                    Object to store in database.
                substrate_generation (scenario.simulation.vne.simulation_objects.SubstrateGenerationSetting, optional):
                    if substrate is saved, pass this along to be able to later
                    retrieve the substrate given the settings.
        """
        gen = None
        if type(object) == networkmodel.PhysicalNetwork:
            if 'substrate_generation' in kwargs:
                gen = kwargs['substrate_generation']

        self.connection.connect()
        with datamodel.db.atomic() as txn:
            network_record = datamodel.Network.create(
                **object.todict()
            )
            for node in object.nodes.itervalues():
                dictionary = node.todict()
                datamodel.Node.create(
                    label=dictionary.pop('label'),
                    network=network_record.network_id,
                    attributes=self.serialize(dictionary)
                )
            for edge in object.edges.itervalues():
                dictionary = edge.todict()
                datamodel.Edge.create(
                    label=dictionary.pop('label'),
                    node_one=dictionary.pop('node_one_id'),
                    node_two=dictionary.pop('node_two_id'),
                    network=network_record.network_id,
                    attributes=self.serialize(dictionary)
                )
            if (type(object) == networkmodel.PhysicalNetwork) and (gen is not None):
                datamodel.NetworkGenerationToNetwork.create(
                    network_generation_setting=gen.identifier,
                    network=network_record.network_id
                )
        object.identifier = network_record.network_id
        self.connection.disconnect()

    def get_object_id(self, network_generation):
        """ Returns object id(s).

            Args:
                network_generation (scenario.simulation.vne.simulation_objects. \
                    NetworkGenerationSetting)

            Returns:
                substrate_ids (list): List of ids.
        """
        self.connection.connect()
        query = datamodel.Network \
            .select() \
            .join(datamodel.NetworkGenerationToNetwork) \
            .where(
                datamodel.NetworkGenerationToNetwork.network_generation_setting == \
                network_generation.identifier
            )
        if query.count() == 0:
            self.connection.disconnect()
            raise custom_exceptions.SubstrateNotKnownError(0, network_generation.todict())
        substrate_ids = [obj.network_id for obj in query.execute()]
        self.connection.disconnect()
        return substrate_ids

    def get_object(self, id):
        def add_node(node_record, network):
            node_record_attributes = self.deserialize(node_record.attributes)
            node_attributes = {
                'label': node_record.label,
                'cpu': node_record_attributes['cpu'],
                'latitude': node_record_attributes['latitude'] if 'latitude' \
                    in node_record_attributes else None,
                'longitude': node_record_attributes['longitude'] if 'longitude' \
                    in node_record_attributes else None,
            }
            network.add_node(**node_attributes)
        def add_edge(edge_record, network):
            edge_record_attributes = self.deserialize(edge_record.attributes)
            edge_attributes = {
                'label': edge_record.edge_id,
                'capacity': edge_record_attributes['capacity'],
                'node_one_id': edge_record.node_one,
                'node_two_id': edge_record.node_two,
                'delay': edge_record_attributes['delay'] if 'delay' \
                    in edge_record_attributes else None,
                'length': edge_record_attributes['length'] if 'length' \
                    in edge_record_attributes else None,
            }
            network.add_edge(**edge_attributes)

        self.connection.connect()
        network_record = datamodel.Network.get(
            datamodel.Network.network_id == id
        )
        if network_record.subclass_type == literals.CLASS_PHYSICAL_NETWORK:
            concrete_network = networkmodel.PhysicalNetwork()
        elif network_record.subclass_type == literals.CLASS_VIRTUAL_NETWORK:
            concrete_network = networkmodel.VirtualNetwork()
        elif network_record.subclass_type == literals.CLASS_NETWORK:
            concrete_network = networkmodel.Network()
        else:
            raise ValueError('Unknown Network subclass stype {}'.format(
                network_record.subclass_type
            ))
        for node in network_record.nodes:
            add_node(node, concrete_network)
        for edge in network_record.edges:
            add_edge(edge, concrete_network)
        self.connection.disconnect()
        concrete_network.identifier = network_record.network_id
        concrete_network.implementor = NetworkImplementor(self.connection)
        return concrete_network

    def update_object(self, **kwargs):
        raise NotImplementedError('Not implemented for class NetworkImplementor')


class EmbeddingImplementor(PeeweeImplementor):

    def __init__(self, db_connection):
        super(EmbeddingImplementor, self).__init__(connection=db_connection)

    def save_object(self, object, **kwargs):
        """ Stores object in database.

            Args:
                object (scenario.simulation.vne.simulation_objects.Embedding):
                    Object to store in database.
        """
        object.embedded_topology.implementor = NetworkImplementor(self.connection)
        object.embedded_topology.save()

        self.connection.connect()
        with self.connection.db.atomic() as txn:
            #network_record = datamodel.Network.create(
            #    **object.embedded_topology.todict()
            #)
            #for node in object.embedded_topology.nodes.itervalues():
            #    dictionary = node.todict()
            #    dictionary['network_id'] = network_record.network_id
            #    datamodel.Node.create(**dictionary)
            #for edge in object.embedded_topology.edges.itervalues():
            #    dictionary = edge.todict()
            #    dictionary['network_id'] = network_record.network_id
            #    datamodel.Edge.create(**dictionary)

            dictionary = object.todict()
            dictionary['network_id'] = object.embedded_topology.identifier
            embedding_record = datamodel.Embedding.create(**dictionary)

            for node_embedding in object.node_embeddings:
                # TODO: adapt node embedding creationg. Use object attributes instead of dict
                dictionary = node_embedding.todict()
                dictionary['embedding'] = embedding_record.embedding_id
                datamodel.NodeEmbedding.create(**dictionary)
            for edge_embedding in object.edge_embeddings:
                # TODO: adapt edge embedding creationg
                dictionary = edge_embedding.todict()
                dictionary['embedding'] = embedding_record.embedding_id
                record = datamodel.EdgeEmbedding.create(**dictionary)
                for node_one, node_two in edge_embedding.physical_edges:
                    dictionary = {
                        'physical_node_one': node_one,
                        'physical_node_two': node_two,
                        'edge_embedding': record.edge_embedding_id
                    }
                    datamodel.EdgeEmbeddingPart.create(**dictionary)
        object.identifier = embedding_record.embedding_id
        self.connection.disconnect()


class SubstrateStateImplementor(PeeweeImplementor):

    def __init__(self, db_connection):
        super(SubstrateStateImplementor, self).__init__(connection=db_connection)

    def save_object(self, object, **kwargs):
        """ Stores object in database.

            Args:
                object (scenario.simulation.vne.simulation_objects.SubstrateState):
                    Object to store in database.
                event_occurrence (scenario.simulation.vne.simulation_objects.EventOccurrence):
                    occurrence after which state has been taken.
        """
        assert 'event_occurrence' in kwargs, 'EventOccurrence object required'
        event_occurrence = kwargs['event_occurrence']
        object.inverse_network.implementor = NetworkImplementor(self.connection)
        object.inverse_network.save()

        self.connection.connect()
        with self.connection.db.atomic() as txn:
            #network_record = datamodel.Network.create(
            #    **object.inverse_network.todict()
            #)
            #for node in object.inverse_network.nodes.itervalues():
            #    dictionary = node.todict()
            #    dictionary['network_id'] = network_record.network_id
            #    datamodel.Node.create(**dictionary)
            #for edge in object.inverse_network.edges.itervalues():
            #    dictionary = edge.todict()
            #    dictionary['network_id'] = network_record.network_id
            #    datamodel.Edge.create(**dictionary)
            dictionary = object.todict()
            dictionary['network'] = object.inverse_network.identifier
            dictionary['event_occurrence'] = event_occurrence.identifier
            record = datamodel.SubstrateState.create(**dictionary)
        object.identifier = record.substrate_state_id
        self.connection.disconnect()


class StochasticProcessImplementor(PeeweeImplementor):

    def get_object_id(self, **kwargs):
        self.connection.connect()
        restrictions = self._assemble_restrictions(kwargs, datamodel.ProcessSetting)
        query = datamodel.ProcessSetting.select().where(*restrictions)
        count = query.count()

        if count == 0:
            self.connection.disconnect()
            raise custom_exceptions.StochasticProcessNotKnownError(kwargs)
        elif count == 1:
            record_id = query.first().process_setting_id
            self.connection.disconnect()
        else:
            self.connection.disconnect()
            raise RuntimeError('Found more than one ProcessSettings record'
                               'for settings {}.'.format(str(kwargs)))
        return record_id

    def get_object(self, identifier):
        self.connection.connect()
        record = datamodel.ProcessSetting.get(
            datamodel.ProcessSetting.process_setting_id == identifier
        )
        if record.type == literals.PROCESS_TYPE_ARRIVAL:
            process = simulation_objects.ArrivalProcess(
                distribution=record.distribution,
                arrival_rate=record.arrival_rate,
                identifier=record.process_setting_id,
                num_requests=record.num_requests
            )
        elif record.type == literals.PROCESS_TYPE_SERVICE:
            process = simulation_objects.ServiceProcess(
                distribution=record.distribution,
                arrival_rate=record.arrival_rate,
                identifier=record.process_setting_id
            )
        else:
            self.connection.disconnect()
            raise ValueError('Unknown process type {}'.format(record.type))
        self.connection.disconnect()
        process.implementor = StochasticProcessImplementor(self.connection)
        return process

    def save_object(self, object, **kwargs):
        """ Saves object and returns database identifier.

            object (scenario.simulation.vne.simulation_objects.StochasticProcess):
                Object to store in database.
        """
        record = datamodel.ProcessSetting.create(**object.todict())
        object.identifier = record.process_setting_id

    def exists_object(self, identifier=None, type=None, arrival_rate=None, distribution=None):
        """ Checks whether object exists.

            Args:
                identifier (int, optional): database id
                type (string, optional): subclass type of process
                arrival_rate (float, optional): Arrival rate
                distribution (string, optional): Type of distribution
        """
        exists = False
        restrictions = []
        if identifier is not None:
            try:
                datamodel.ProcessSetting.get(datamodel.ProcessSetting.process_setting_id == identifier)
                exists = True
            except datamodel.ProcessSetting.DoesNotExist:
                pass
        else:
            if type is not None:
                restrictions.append(datamodel.ProcessSetting.type % type)
            if arrival_rate is not None:
                restrictions.append(datamodel.ProcessSetting.arrival_rate == arrival_rate)
            if distribution is not None:
                restrictions.append(datamodel.ProcessSetting.distribution % distribution)

            self.connection.connect()
            query = datamodel.EventGenerationSetting.select().where(*restrictions)
            count = query.count()

            if count > 0:
                exists = True
        self.connection.disconnect()
        return exists


class NetworkGenerationSettingsImplementor(PeeweeImplementor):

    def get_object_id(self, **kwargs):
        self.connection.connect()
        restrictions = self._assemble_restrictions(kwargs, datamodel.NetworkGenerationSetting)

        query = datamodel.NetworkGenerationSetting.select().where(*restrictions)
        count = query.count()
        if count == 0:
            self.connection.disconnect()
            raise custom_exceptions.NetworkGenerationSettingsNotKnownError(kwargs)
        elif count == 1:
            vnr_record = query.first().network_generation_setting_id
            self.connection.disconnect()
        else:
            self.connection.disconnect()
            raise RuntimeError('Found more than one VnrGenerationSettings record'
                               'for settings {}.'.format(str(kwargs)))
        return vnr_record

    def save_object(self, object, **kwargs):
        """ Creates record and returns database identifier

            Args:
                object (scenario.simulation.vne.simulation_objects.NetworkGenerationSettings):
                    Object to store in database.
        """
        record = datamodel.NetworkGenerationSetting.create(**object.todict())
        object.identifier = record.network_generation_setting_id

    def get_object(self, object_id):
        self.connection.connect()
        record = datamodel.NetworkGenerationSetting.get(
            datamodel.NetworkGenerationSetting.network_generation_setting_id == object_id
        )
        obj = simulation_objects.NetworkGenerationSettings(
            model=record.model,
            connectivity=record.connectivity,
            m=record.m,
            m0=record.m0,
            alpha=record.alpha,
            beta=record.beta,
            minx=record.minx,
            maxx=record.maxx,
            miny=record.miny,
            maxy=record.maxy,
            order=record.order,
            min_order=record.min_order,
            max_order=record.max_order,
            min_capacity=record.min_capacity,
            max_capacity=record.max_capacity,
            capacity_generation=record.capacity_generation,
            min_cpu=record.min_cpu,
            max_cpu=record.max_cpu,
            min_edge_distance=record.min_edge_distance,
            max_edge_distance=record.max_edge_distance,
            delay_per_km=record.delay_per_km,
            is_substrate=record.is_substrate
        )
        obj.identifier = record.network_generation_setting_id
        self.connection.disconnect()
        obj.implementor = NetworkGenerationSettingsImplementor(self.connection)
        return obj


class EventGenerationSettings(PeeweeImplementor):

    def get_object_id(self, **kwargs):
        """ Retrieves the record for the specified setting ids. If
            the record does not exist it is created.

            Args:
                service_process (simulation object)
                arrival_process (simulation object)
                network_generation_process (simulation object)

            Returns:
                event_generation_record (int): Database ID of respective record

            Raises:
                RuntimeError if record could not be created or more than one
                    record was found.
        """
        service_process_id = kwargs['service_process'].identifier
        arrival_process_id = kwargs['arrival_process'].identifier
        vnr_generation_process_id = kwargs['network_generation_process'].identifier

        self.connection.connect()
        query = (
            datamodel.EventGenerationSetting.select().where(
                datamodel.EventGenerationSetting.network_setting == vnr_generation_process_id
            )
        ).execute()
        ids = [record.event_generation_setting_id for record in query]

        if len(ids) > 0:
            query = (
                datamodel.ProcessToEventSetting.select().where(
                    datamodel.ProcessToEventSetting.event_generation_setting << ids,
                    datamodel.ProcessToEventSetting.process_setting == arrival_process_id
                )
            ).execute()
            ids = [record.event_generation_setting_id for record in query]
        else:
            count = 0

        if len(ids) > 0:
            query = (
                datamodel.ProcessToEventSetting.select().where(
                    datamodel.ProcessToEventSetting.event_generation_setting << ids,
                    datamodel.ProcessToEventSetting.process_setting == service_process_id
                )
            )
            count = query.count()
        else:
            count = 0

        if count == 0:
            raise custom_exceptions.EventGenerationSettingsNotKnownError(
                service_process_id, arrival_process_id, vnr_generation_process_id
            )
        elif count == 1:
            record = query.first().event_generation_setting_id
        else:
            raise RuntimeError('Found more than one EventGenerationSettings record'
                               'for settings {} {} {}.'.format(
                arrival_process_id, service_process_id, vnr_generation_process_id)
            )
        return record

    def get_object(self, identifier):
        self.connection.connect()
        evtgen_record = datamodel.EventGenerationSetting.get(
            datamodel.EventGenerationSetting.event_generation_setting_id == identifier
        )
        query = (
            datamodel.ProcessToEventSetting
                .select()
                .join(datamodel.EventGenerationSetting)
                .where(datamodel.EventGenerationSetting.event_generation_setting_id == identifier)
        )
        count = query.count()
        if count != 2:
            raise RuntimeError((
                'Expected two process entries for event generation setting'
                '{}, but found {} instead').format(identifier, count)
            )
        process_ids = [record.process_setting_id for record in query.execute()]
        self.connection.disconnect()

        simp = StochasticProcessImplementor(self.connection)
        if simp.exists_object(process_ids[0], type=literals.PROCESS_TYPE_ARRIVAL):
            arrival_process = simp.get_object(process_ids[0])
            service_process = simp.get_object(process_ids[1])
        else:
            arrival_process = simp.get_object(process_ids[1])
            service_process = simp.get_object(process_ids[0])

        network_settings = NetworkGenerationSettingsImplementor(self.connection) \
            .get_object(evtgen_record.network_setting.network_generation_setting_id)

        settings = simulation_objects.EventGenerationSettings(
            arrival_process=arrival_process,
            service_process=service_process,
            network_generation_settings=network_settings,
            identifier=evtgen_record.event_generation_setting_id,
            implementor=EventGenerationSettings(self.connection)
        )
        settings.identifier = evtgen_record.event_generation_setting_id
        settings.implementor = EventGenerationSettings(self.connection)
        return settings

    def save_object(self, object, **kwargs):
        """ Save object to database.

            Args:
                object (scenario.simulation.vne.simulation_objects.EventGenerationSetting):
                    object to be saved.
        """
        self.connection.connect()
        record = datamodel.EventGenerationSetting.create(
            network_setting=object.network_generation_settings.identifier
        )
        datamodel.ProcessToEventSetting.create(
            process_setting=object.arrival_process.identifier,
            event_generation_setting=record.event_generation_setting_id,
        )
        datamodel.ProcessToEventSetting.create(
            process_setting=object.service_process.identifier,
            event_generation_setting=record.event_generation_setting_id,
        )
        self.connection.disconnect()
        object.identifier = record.event_generation_setting_id


class EventHeapImplementor(PeeweeImplementor):

    def save_object(self, object, **kwargs):
        """ Saves event heap record to database.

            Args:
                object (scenario.simulation.vne.simulation_objects.EventQueue):
                    New Event Queue for which record should be created.
                event_generation (scenario.simulation.vne.simulation_objects.EventGenerationSettings):
                    Object reprsenting the for the events used settings.

            Returns:
                database_id (int)

            Raises:
                AssertionError if one of the required additional arguments is
                    not present.
        """
        assert 'event_generation' in kwargs, 'EventGeneration setting object expected'
        event_generation = kwargs['event_generation']
        self.connection.connect()
        event_queue_record = datamodel.EventHeap.create(
            event_generation_setting=event_generation.identifier
        )
        object.identifier = event_queue_record.event_heap_id
        self.connection.disconnect()

    def get_object_id(self, event_generation_id):
        self.connection.connect()

        heaps = datamodel.EventHeap.select().where(
            datamodel.EventHeap.event_generation_setting == event_generation_id
        )
        heap_ids = [heap.event_heap_id for heap in heaps.execute()]
        self.connection.disconnect()
        if len(heap_ids) == 0:
            raise custom_exceptions.EventHeapNotKnownError(
                'Could not find event heap for event generation id {}'.format(event_generation_id))
        return heap_ids

    def get_object(self, id, **kwargs):
        """ Get a concrete event queue.

            Args:
                 id (int): database id of event heap.

            Returns (scenario.simulation.vne.simulation_objects.EventQueue)
        """
        self.connection.connect()
        heap_record = datamodel.EventHeap.get(datamodel.EventHeap.event_heap_id == id)
        concrete_heap = simulation_objects.EventQueue(identifier=heap_record.event_heap_id)
        self.connection.disconnect()

        for event in heap_record.events:
            network = NetworkImplementor(self.connection).get_object(event.network.network_id)
            concrete_heap.heappush(simulation_objects.ArrivalEvent(
                identifier=event.event_id,
                arrival_time=event.time,
                lifetime=event.lifetime,
                virtual_network=network
            ))
        concrete_heap.identifier = heap_record.event_heap_id
        concrete_heap.implementor = EventHeapImplementor(self.connection)
        return concrete_heap


class OccurredEventImplementor(PeeweeImplementor):

    def save_object(self, object, **kwargs):
        self.connection.connect()
        record = datamodel.EventOccurrence.create(
            time=object.time,
            subclass_type=object.subclass_type,
            occurred=1,
            event_id=object.event_id,
            embedding=object.embedding.identifier,
            run_execution=object.run_execution.identifier
        )
        object.identifier = record.event_occurrence_id
        self.connection.disconnect()


class LearningModelImplementor(PeeweeImplementor):

    def get_object_id(self, **kwargs):
        path = kwargs.pop('serialized_model')
        self.connection.connect()
        model = datamodel.LearningModel.select().where(
            datamodel.LearningModel.serialized_model == path
        ).first()
        if model is None:
            #logger.info('No model found for path {}'.format(path))
            learning_model_id = -1
        else:
            learning_model_id = model.learning_model_id
        self.connection.disconnect()

        return learning_model_id

    def get_object(self, id):
        self.connection.connect()
        model = datamodel.LearningModel.get(
            datamodel.LearningModel.learning_model_id == id
        )
        concrete = simulation_objects.BrezeSupervisedRnnModel(
            identifier=model.learning_model_id,
            implementor=self.connection,
            modelpath=model.serialized_model,
            **self.deserialize(model.aux_params)
        )
        self.connection.disconnect()
        return concrete


class ArrivalEventImplementor(PeeweeImplementor):
    def save_object(self, object, **kwargs):
        assert 'event_queue' in kwargs
        event_queue = kwargs['event_queue']
        self.connection.connect()
        record = datamodel.Event.create(
            network=object.virtual_network.identifier,
            time=object.arrival_time,
            lifetime=object.lifetime,
            event_heap=event_queue.identifier
        )
        object.identifier = record.event_id
        self.connection.disconnect()


class AlgorithmSettingImplementor(PeeweeImplementor):

    def get_object_id(self, name, parameter):
        """ Return id of record.

            Args:
                name (string): Name of algorithm.
                parameter (dict): Additional parameter.
        """
        self.connection.connect()
        query = datamodel.AlgorithmSetting.select().where(
            datamodel.AlgorithmSetting.name % name
        )
        found_algos = []
        for record in query.execute():
            accept = True
            record_params = self.deserialize(record.parameter)
            for key, value in parameter.iteritems():
                if str(key) in record_params:
                    dbvalues = [str(record_params[str(key)])]
                    try:
                        dbvalues.append(float(dbvalues[0]))
                    except Exception:
                        pass
                    if value in dbvalues:
                        continue
                    else:
                        accept = False
                        break
                else:
                    accept = False
                    break
            if accept:
                found_algos.append(record)
            else:
                continue

        if len(found_algos) == 0:
            self.connection.disconnect()
            raise custom_exceptions.AlgorithmNotKnownError(
                'No algorithm found with name {} and parameter {}'.format(name, str(parameter))
            )
        elif len(found_algos) == 1:
            record_id = found_algos[0].algorithm_setting_id
            self.connection.disconnect()
        else:
            self.connection.disconnect()
            raise RuntimeError(
                'More than one algorithm found with name {} and parameter {}'.format(
                    name, str(parameter)
                ))
        return record_id

    def get_object(self, identifier):
        """ Returns simulation object.

            Args:
                identifier (int): database id of object.

            Returns scenario.simulation.vne.simulation_object.Algorithm
        """
        self.connection.connect()
        record = datamodel.AlgorithmSetting.get(
            datamodel.AlgorithmSetting.algorithm_setting_id == identifier
        )
        algo = simulation_objects.Algorithm(
            name=record.name,
            parameter=self.deserialize(record.parameter),
            identifier=record.algorithm_setting_id
        )
        self.connection.disconnect()
        algo.implementor = AlgorithmSettingImplementor(self.connection)
        return algo

    def save_object(self, object, **kwargs):
        """ Saves object to database.

            Args:
                object (scenario.simulation.vne.simulation_objects.Algorithm):
                    Object to be stored in database.
        """
        self.connection.connect()
        record = datamodel.AlgorithmSetting.create(
            name=object.name,
            parameter=self.serialize(object.parameter)
        )
        object.identifier = record.algorithm_setting_id
        self.connection.disconnect()


class ScenarioImplementor(PeeweeImplementor):

    def exists_object(self, event_generation, network_generation,
                      learning_model_id, algorithm_setting):
        try:
            datamodel.Scenario.get(
                datamodel.Scenario.event_generation_setting == event_generation.identifier,
                datamodel.Scenario.network_generation_setting == network_generation.identifier,
                datamodel.Scenario.learning_model == learning_model_id,
                datamodel.Scenario.algorithm_setting == algorithm_setting.identifier
            )
            ret = True
        except datamodel.Scenario.DoesNotExist:
            ret = False
        return ret

    def get_object_id(self, event_generation, network_generation,
                      learning_model_id, algorithm_setting):
        try:
            record = datamodel.Scenario.get(
                datamodel.Scenario.event_generation_setting == event_generation.identifier,
                datamodel.Scenario.network_generation_setting == network_generation.identifier,
                datamodel.Scenario.learning_model == learning_model_id,
                datamodel.Scenario.algorithm_setting == algorithm_setting.identifier
            )
        except datamodel.Scenario.DoesNotExist:
            raise custom_exceptions.ScenarioNotKnownError()
        return record

    def save_object(self, object, **kwargs):
        record = datamodel.Scenario.create(
            event_generation_setting=object.event_generation.identifier,
            network_generation_setting=object.network_generation.identifier,
            learning_model=object.learning_model_id,
            algorithm_setting=object.algorithm_setting.identifier
        )
        object.identifier = record.scenario_id

    def get_object(self, object_id):
        self.connection.connect()
        record = datamodel.Scenario.get(
            datamodel.Scenario.scenario_id == object_id
        )
        self.connection.disconnect()
        algorithm_setting = AlgorithmSettingImplementor(self.connection).get_object(
            record.algorithm_setting.algorithm_setting_id
        )
        if record.learning_model is not None:
            learning_model_id = record.learning_model.learning_model_id
        else:
            learning_model_id = -1
        event_generation = EventGenerationSettings(self.connection).get_object(
            record.event_generation_setting.event_generation_setting_id
        )
        network_generation = NetworkGenerationSettingsImplementor(self.connection) \
            .get_object(record.network_generation_setting.network_generation_setting_id)
        scenario = simulation_objects.Scenario(
            algorithm_setting, learning_model_id, network_generation, event_generation
        )
        scenario.identifier = record.scenario_id
        scenario.implementor = ScenarioImplementor(self.connection)
        return scenario


class RunConfigurationImplementor(PeeweeImplementor):

    def exists_object(self, scenario, network, event_queue):
        """ Retrieves object id from database.

        Args:
            scenario (scenario.simulation.vne.simulation_objects.Scenario):
                Scenario run configuration belongs to.
            network (network_model.vne.networkmodel.PhysicalNetwork):
                Substrate network.
            event_queue (scenario.simulation.vne.simulation_objects.EventQueue):
                Event Queue Run is configured to work with.

        Returns:
            None
        """
        self.connection.connect()
        query = datamodel.RunConfiguration.select().where(
            datamodel.RunConfiguration.scenario == scenario.identifier,
            datamodel.RunConfiguration.network == network.identifier,
            datamodel.RunConfiguration.event_heap == event_queue.identifier
        )
        count = query.count()
        self.connection.disconnect()
        if count == 0:
            return False
        elif count == 1:
            return True
        else:
            raise RuntimeError('Found more than one RunConfiguration record')

    def get_object_id(self, scenario, network, event_queue):
        """ Retrieves object id from database.

        Args:
            scenario (scenario.simulation.vne.simulation_objects.Scenario):
                Scenario run configuration belongs to.
            network (network_model.vne.networkmodel.PhysicalNetwork):
                Substrate network.
            event_queue (scenario.simulation.vne.simulation_objects.EventQueue):
                Event Queue Run is configured to work with.

        Returns:
            None
        """
        self.connection.connect()
        query = datamodel.RunConfiguration.select().where(
            datamodel.RunConfiguration.scenario == scenario.identifier,
            datamodel.RunConfiguration.network == network.identifier,
            datamodel.RunConfiguration.event_heap == event_queue.identifier
        )
        count = query.count()
        if count == 0:
            raise custom_exceptions.RunConfigurationNotKnownError()
        elif count == 1:
            record_id = query.first().run_configuration_id
            self.connection.disconnect()
        else:
            self.connection.disconnect()
            raise RuntimeError('Found more than one RunConfiguration record')
        return record_id

    def get_object(self, identifier):
        self.connection.connect()
        record = datamodel.RunConfiguration.get(
            datamodel.RunConfiguration.run_configuration_id == identifier
        )
        self.connection.disconnect()

        network = NetworkImplementor(self.connection).get_object(record.network.network_id)
        event_queue = EventHeapImplementor(self.connection).get_object(
            record.event_heap.event_heap_id
        )
        scenario = ScenarioImplementor(self.connection).get_object(
            record.scenario.scenario_id
        )
        run_object = simulation_objects.RunConfiguration(
            scenario,
            network,
            event_queue,
            record.run_configuration_id
        )
        run_object.implementor = RunConfigurationImplementor(self.connection)
        return run_object

    def save_object(self, object, **kwargs):
        """ Write object to data source

        Args:
            object (scenario.simulation.vne.simulation_objects.RunConfiguration):
                configuration to be stored.
            **kwargs (dict, optional): Additional parameter

        Returns:
            None
        """
        record = datamodel.RunConfiguration.create(
            scenario=object.scenario.identifier,
            network=object.network.identifier,
            event_heap=object.event_queue.identifier
        )
        object.identifier = record.run_configuration_id


class RunExecutionImplementor(PeeweeImplementor):

    def save_object(self, object, **kwargs):
        """ Save object to database.
        """
        run_configuration_id = kwargs.pop('run_configuration_id')
        self.connection.connect()
        attributes = object.todict()
        record = datamodel.RunExecution.create(
            run_configuration=run_configuration_id,
            **attributes
        )
        object.identifier = record.run_execution_id
        self.connection.disconnect()

    def update_object(self, object, **kwargs):
        """ update object in database.

            Args:
                object (scenario.simulation.vne.simulation_objects.RunExecution):
                    Object to update

            Returns:
                None
        """
        self.connection.connect()
        q = datamodel.RunExecution.update(
            stage_of_execution=object.stage_of_execution,
            num_successful_embeddings=object.num_successful_embeddings,
            num_failed_embeddings=object.num_failed_embeddings,
            num_infeasible_embeddings=object.num_infeasible_embeddings,
            execution_time=object.execution_time,
            num_cores=object.num_cores,
            priority=object.priority,
            num_filtered_embeddings=object.num_filtered_embeddings
        ).where(
            datamodel.RunExecution.run_execution_id == object.identifier
        )
        q.execute()
        self.connection.disconnect()


class ObjectFactory(object):

    @classmethod
    def produce(cls, object_class, connection):
        if object_class == networkmodel.PhysicalNetwork:
            return NetworkImplementor(connection)
        elif object_class == networkmodel.VirtualNetwork:
            return NetworkImplementor(connection)
        elif object_class == networkmodel.Network:
            return NetworkImplementor(connection)
        elif object_class == simulation_objects.EventQueue:
            return EventHeapImplementor(connection)
        elif object_class == simulation_objects.ArrivalEvent:
            return ArrivalEventImplementor(connection)
        elif object_class == simulation_objects.StochasticProcess:
            return StochasticProcessImplementor(connection)
        elif object_class == simulation_objects.NetworkGenerationSettings:
            return NetworkGenerationSettingsImplementor(connection)
        elif object_class == simulation_objects.Algorithm:
            return AlgorithmSettingImplementor(connection)
        elif object_class == simulation_objects.Scenario:
            return ScenarioImplementor(connection)
        elif object_class == simulation_objects.EventGenerationSettings:
            return EventGenerationSettings(connection)
        elif object_class == simulation_objects.BrezeSupervisedRnnModel:
            return LearningModelImplementor(connection)
        elif object_class == simulation_objects.RunConfiguration:
            return RunConfigurationImplementor(connection)
        elif object_class == simulation_objects.RunExecution:
            return RunExecutionImplementor(connection)
        elif object_class == simulation_objects.EventOccurrence:
            return OccurredEventImplementor(connection)
        elif object_class == simulation_objects.Embedding:
            return EmbeddingImplementor(connection)
        elif object_class == simulation_objects.SubstrateState:
            return SubstrateStateImplementor(connection)
        else:
            raise ValueError('Unregistered object class {}'.format(object_class))


