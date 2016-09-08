class ProblemNotKnownError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class NoConfigFileFoundError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class StorageTypeNotKnownError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class InterfaceNotConfiguredError(Exception):
    def __int__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class InterfaceNotKnownError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class StrategyNotKnownError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class AlgorithmNotKnownError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class RawScenarioSourceNotKnown(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class SubstrateNotKnownError(Exception):

    def __init__(self, order, model):
        """ Initializes object.

            Args:
                order (int): Order of substrate.
                model (string): Type of network.
        """
        super(SubstrateNotKnownError, self).__init__(
            'No substrate found for order {} and model {}'.format(order, model)
        )


class EventHeapNotKnownError(Exception):
    pass


class StochasticProcessNotKnownError(Exception):
    def __init__(self, parameter):
        super(StochasticProcessNotKnownError, self).__init__(
            'No Process with parameter {} exists'.format(str(parameter))
        )


class NetworkGenerationSettingsNotKnownError(Exception):
    def __init__(self, parameter):
        super(NetworkGenerationSettingsNotKnownError, self).__init__(
            'No settings with parameter {} exists'.format(str(parameter))
        )


class EventGenerationSettingsNotKnownError(Exception):
    def __init__(self, service_process_id, arrival_process_id,
                 vnr_generation_process_id):
        super(EventGenerationSettingsNotKnownError, self).__init__(
            (
                'No event generation settings with parameter '
                'arrival process {}, service process {} and'
                'network generation settings {} exists'
            ).format(
                    service_process_id, arrival_process_id, vnr_generation_process_id
            )
        )


class ScenarioNotKnownError(Exception):
    pass


class RunConfigurationNotKnownError(Exception):
    pass


class LearningModelNotKnownError(Exception):
    pass
