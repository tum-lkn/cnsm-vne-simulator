from peewee_implementors import ObjectFactory
from input_output.data_interface.bridge.datamodel import ConnectionManager

class ImplementorFactory(object):
    @classmethod
    def produce(cls, interface, object_class):
        if type(interface) == ConnectionManager:
            return ObjectFactory.produce(
                object_class,
                interface
            )
        else:
            raise ValueError('Unknown Type {}'.format(str(interface)))
