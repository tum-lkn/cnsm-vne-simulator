import celery
import celery_configuration
import scenario.simulation.concrete_simulations_factories

"""
Contains the interface to use celery
"""

__authors__ = 'Johannes Zerwas'

app = celery.Celery()
app.config_from_object(celery_configuration)


@app.task
def do_simulation(simulation_config, if_input, if_output):
    """
    Starts a simulation via Celery
    Args:
        simulation_config:  Dict that contains the information required for a simulation, i.e., problem type,
            setup id, input set id and simulation strategy
        if_input:   Input interface
        if_output:  List of output interfaces

    Returns:
        0 if run failed
        1 if run succeeded
    """
    sim = scenario.simulation.concrete_simulations_factories.SimulationFactory.produce(simulation_config, if_input,
                                                                                       if_output)
    return sim.run()
