import modifiedmelo
import filtering
import literals


class AlgorithmFactory(object):
    @classmethod
    def produce(self, name, substrate, **args):
        """ Returns one of registered algorithm objects

            Args:
                name (string): Name of algorithm.
                substrate (fnss.Topology, scenario.simulation.vne.helpers.VnetTumWrapper):
                    substrate used in algorithm.
                args (dict): Additional parameters. Will be passed to algorithm.

            Returns:
                algorithm, anytype
        """
        legacy_map = {
            'melo_sdp': 'SDP',
            'melo_lb': 'LB'
        }
        if name in legacy_map:
            name = legacy_map[name]

        if name == literals.ALGORITHM_MELO_SDP:
            return modifiedmelo.MeloSDP(physicalNetwork=substrate, **args)
        elif name == literals.ALGORITHM_MELO_LB:
            return modifiedmelo.MeloLB(physicalNetwork=substrate, **args)
        elif name == literals.ALGORITHM_RNN_FILTER:
            return filtering.RnnFilter(**args)
        else:
            raise ValueError('Unkown algorithm name {}'.format(name))
