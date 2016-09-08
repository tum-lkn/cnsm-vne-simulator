import input_output.data_interface.bridge.datamodel as vne_interface
import errors.homebrewederrors as errorbrewer
import literals

__authors__ = 'Johannes Zerwas, Patrick Kalmbach, Andreas Blenk'


class InterfaceFactory(object):
    @classmethod
    def produce(cls, problem_type, interface_config=None):
        """

        Args:
            problem_type (string): Type of problem.
            interface_config (input_output.data_interface.interface_config.DataInterfaceConfig):
                subclass of this.

        Returns:
            interface

        """
        if problem_type == 'vne':
            if interface_config.storage_type == literals.STORAGE_PEEWEE:
                interface = vne_interface.ConnectionManager(
                    input_config=interface_config
                )
                #interface.connect()
            else:
                raise errorbrewer.StorageTypeNotKnownError('storage type is unknown')
        # ===============================
        # Interface factory does not know
        # ===============================
        else:
            raise errorbrewer.ProblemNotKnownError('InterfaceFactory knows: cpp,hpp,dhpp,vne')
        return interface
