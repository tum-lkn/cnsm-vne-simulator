# import input_output.data_interface.interface_factory as if_factory
# import input_output.data_interface.interface_config as if_config_factory
#
# __author__ = "Andreas Blenk"
#
#
# class DataInterfaceFactory(object):
#     @classmethod
#     def produce(cls, if_configuration_as_dict):
#         if_configuration = if_config_factory.InterfaceConfigFactory.produce(if_configuration_as_dict)
#         return if_factory.InterfaceFactory.produce('cpp',
#                                                    if_configuration)
