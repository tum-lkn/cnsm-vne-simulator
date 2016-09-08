"""

"""
import re
import literals
__authors__ = 'Patrick Kalmbach'


class BaseStrategy(object):
    """ A strategy governs the write behaviour of the simulation. To decide the
        time when to write to the data sinks, The amount of occurred events,
        i,e, the length of `simulation_steps` is used.
    """
    def __init__(self):
        """ Initializes object.

            Args:
                run_execution (scenario.simulation.vne.simulation_objects.RunExecution):
                    Run the simulation represents
        """
        self.simulation_steps = []
        self.run_execution = None  # Run execution changes constantly,
                                            # makes no sense to add it to update
                                            # list every time, so pass reference
                                            # and do update only once

    def append_simulation_step(self, simulation_step):
        """ Add simulation step object to list.

            Args:
                simulation_step (scenario.simulation.vne.simulation_objects.SimulationStep):
                    simulation step object that needs to be created.
        """
        raise NotImplementedError

    def write(self):
        """ Write all objects to the database.
        """
        self.run_execution.update()

        for step in self.simulation_steps:
            if step.embedding is not None:
                step.embedding.save()
            step.event_occurrence.save()
            step.substrate_state.save(event_occurrence=step.event_occurrence)


class WriteAfterRunStrategy(BaseStrategy):
    def __init__(self):
        super(WriteAfterRunStrategy, self).__init__()


    def append_simulation_step(self, simulation_step):
        """ Add simulation step object to list.

            Args:
                simulation_step (scenario.simulation.vne.simulation_objects.SimulationStep):
                    simulation step object that needs to be created.
        """
        self.simulation_steps.append(simulation_step)


class IntermediateWriteStrategy(BaseStrategy):
    def __init__(self, write_after):
        super(IntermediateWriteStrategy, self).__init__()
        self._write_after = write_after

    def append_simulation_step(self, simulation_step):
        """ Add simulation step object to list.

            Args:
                simulation_step (scenario.simulation.vne.simulation_objects.SimulationStep):
                    simulation step object that needs to be created.
        """
        self.simulation_steps.append(simulation_step)
        if len(self.simulation_steps) >= self._write_after:
            self.write()


class StrategyFactory(object):
    @classmethod
    def produce(cls, simulation_strategy):
        if simulation_strategy == literals.WRITE_AT_END_STRATEGY:
            return WriteAfterRunStrategy()
        elif re.match(literals.WRITE_AFTER_STRATEGY, simulation_strategy) is not None:
            re.match(literals.WRITE_AFTER_STRATEGY, simulation_strategy)
            number = int(re.sub('\D', '', simulation_strategy))
            return IntermediateWriteStrategy(number)
        else:
            raise KeyError('Strategy not found. Strategy {} not found for vne problem'.format(
                simulation_strategy
            ))
