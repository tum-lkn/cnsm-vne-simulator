import json
import logging

class AbstractImplementor(object):

    def __init__(self, connection):
        self.connection = connection
        self.substrate_map = {
            'ER': 'erdos_renyi_topology',
            'BA': 'ba_topology',
            'STAR': 'star_topology',
            'FM':  'full_mesh_topology',
            'WAX': 'waxman_2_topology'
        }
        self.network_map = {
            'ER': 'erdoes_renyi',
            'BA': 'ba',
            'STAR': 'star',
            'FM': 'fm',
            'WAX': 'wax'
        }
        self.logger = logging.getLogger(str(self.__class__))
        self.logger.setLevel(logging.DEBUG)

    def serialize(self, element):
        return unicode(json.dumps(element))

    def deserialize(self, element):
        if element.find("u'") == -1:
            element = str(element).replace("'", '"')
        else:
            element = str(element).replace("u'", '"')
            element = element.replace("'", '"')
        element = element.replace('[]', '0')
        return json.loads(element)

    def save_object(self, object, **kwargs):
        raise NotImplementedError

    def update_object(self, object, **kwargs):
        raise NotImplementedError

    def get_object(self, **kwargs):
        raise NotImplementedError

    def get_object_id(self, **kwargs):
        raise NotImplementedError

    def exists_object(self, **kwargs):
        raise NotImplementedError




