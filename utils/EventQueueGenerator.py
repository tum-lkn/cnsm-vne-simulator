""" Implements an Generator for EventQueues for VNE scenarios
"""
from NetworkGenerator import NetworkGenerator
import network_model.networkmodel as vne_networkmodel
import scenario.simulation.simulation_objects as sim_objects
import numpy as np
import scipy.stats as stats
import time

class EventGenerator(object):
    """
    Generates the different events (`VnrArrivalEvent`, `VnrDepartureEvent`).
    Has the possibility to generate VNRs from different models following a
    uniform or custom discrete distribution. See description of parameter
    `model` for constructor for details.
    """

    def __init__(self, lmbd, avg_life_time, model=None, num_requests=None,
                 experiment_duration=None, seed=None):
        """
        Initialize EventGenerator object.

        Args:
            num_requests (int): Number of arriving VNRs during this time step
            model (String, List, Dictionary): Type of topology vnrs should follow.
                For a specific type, use a String, for multiple types use either
                a List or a Dictionary:
                    * List: Uniform probability for any type.
                    * Dictionary: Mapping String --> probability of selecting
                        model type (probabilities must sum to one).

        """
        self._model = model
        self._lambda = int(lmbd)
        self._avg_life_time = float(avg_life_time)
        self._seed = int((time.time() * 10) % 4294967295) if seed is None else float(seed)
        if experiment_duration is not None:
            tmp_num_requests = int(float(experiment_duration) / 100. * lmbd)
            if tmp_num_requests < num_requests:
                self._num_requests = int(tmp_num_requests)
            else:
                self._num_requests = int(num_requests)
        else:
            self._num_requests = int(num_requests)

    def determine_model(self):
        """
        Create List of names of network models to use for network generation.

        Returns:
            ret: List of Strings.

        Raises:
            AssertionError if either `self._num_requests` or `self._model` is None

        """
        assert self._num_requests is not None
        assert self._model is not None

        if type(self._model) == str:
            ret = [self._model for i in range(int(self._num_requests))]
        elif type(self._model) == unicode:
            ret = [str(self._model) for i in range(int(self._num_requests))]
        elif type(self._model) == list:
            ret = []
            for i in range(self._num_requests):
                ret.append(self._model[int(np.random.uniform(0, len(self._model)))])
        else:
            vals = []
            probs = []
            for key, val in self._model.iteritems():
                vals.append(key)
                probs.append(probs)
            rv = stats.rv_discrete(values=(len(vals), probs))
            samples = rv.rvs(size=self._num_requests)
            ret = [vals[i] for i in samples]
        return ret

    def generate_networks(self, **kwargs):
        models = self.determine_model()
        max_num_nodes = kwargs.pop('max_order')
        min_num_nodes = kwargs.pop('min_order')

        if 'implementor' in kwargs:
            implementor = kwargs.pop('implementor')
        else:
            implementor = None
        vnrs = []

        for i in range(self._num_requests):
            kwargs['order'] = int(np.random.uniform(min_num_nodes, max_num_nodes))
            generator = NetworkGenerator.method_factory(models[i])
            vnr = generator(**kwargs)
            vnr = vne_networkmodel.VirtualNetwork(fnss_topology=vnr)
            if implementor is not None:
                vnr.implementor = implementor
            vnrs.append(vnr)
        return vnrs

    def generate_event_times(self):
        """
        Model the waiting time between requests. This is done via an
        exponential distribution, as:

        Let the arrival of requests be distributed according to Poi(\lambda),
        then the probability of no request arrived after t time steps is:
        ..math::

              P(no arrivals at t) &= (t \lamda)^{0}\frac{e^{-t \lambda}{0!}\\
                                  &= e^{-t lambda}

        The probability of having at least one event in tis time period is:

            ..math:: P(Waiting time < t) = P(W < t) = 1 - e^{-t \lambda}

        This is the CDF of an exponential distribution. Taking the derivative
        after t we get the PDF with:

            ..math:: p(t) = \lambda e^{-t \lambda}

        If we have an average arrival of 5 Requests per 100 time units we
        have:

            ..math:: \lambda = \frac{5}{100}

        Numpy parameterizes the exponential distribution with the precision,
        i.e. the inverse, thus we get:

            ..math:: \beta = \left(\frac{5}{100}\right)^{-1} = \frac{100}{5}=20
        """
        waiting_mean = 1. / (self._lambda / 100.)
        arrival_time = 0
        times = []
        random_state = np.random.RandomState(seed=self._seed)

        for i in range(self._num_requests):
            lifetime = random_state.exponential(self._avg_life_time)
            arrival_time += np.random.exponential(waiting_mean)
            times.append((arrival_time, lifetime))
        return times

    def generate_events(self, vnrs, times):
        pairs = zip(times, vnrs)
        queue = []
        for (time, lifetime), vnr in pairs:
            event = sim_objects.ArrivalEvent(
                arrival_time=time,
                lifetime=lifetime,
                virtual_network=vnr
            )
            queue.append(event)
        return queue
