import cli
import os
import sys
import logging
import logging.config
import yaml

import configurator
import input_output.data_interface.interface_factory as if_factory
import errors.homebrewederrors
import scenario.concrete_scenarios_factories as sc_factory

__authors__ = "Johannes Zerwas, Andreas Blenk, Michael Manhart"


class Control(object):
    """
    Manages the configuration, logging and basic intialization of the framework.
    Attributes:
        scenarios: A list of instances of scenario.scenario.RawScenario (or inherited classes) that contain the
            scenarios read from the configuration file
        cliparser: Arguments parsed from the comment line to overwrite/extend the configuration file

    """

    def __init__(self,
                 configurationfile=None,
                 loggingfile=None):
        self.configfile = configurationfile
        self.loggingfile = loggingfile

        self.scenarios = []
        self.cliparser = None
        self.config = None
        self.input = None
        self.output = None
        self.logger = None
        self.input_config = None
        self.output_configurations = []

    def create_cli_parser(self):
        self.cliparser = cli.create_cl_argparser()

    def create_configuration(self):
        """
        Create configuration dependent on where to get information from ...
        Returns:
            None
        """
        if (self.cliparser is not None) and (len(sys.argv) > 1):
            # Check if args are actually given to the parser
            args = self.cliparser.parse_args()
        else:
            args = None
        self.config = configurator.Config(self.configfile, args)

    def create_logging(self):
        """
        Initializes the logging based on the logging configuration given in the config file
        Returns:
            None
        """

        if self.config is not None:
            self.loggingfile = self.config.get_logging_default_path()
            path = self.loggingfile
        elif self.loggingfile is not None:
            path = self.loggingfile
        else:
            raise errors.homebrewederrors.NoConfigFileFoundError("No logging file set!")

        # Check if path really exists!
        value = os.getenv(self.config.get_logging_env_key(), None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                log_config = yaml.load(f)
            logging.config.dictConfig(log_config)
        elif self.config is not None:
            logging.basicConfig(level=self.config.get_logging_default_level())
        else:
            raise errors.homebrewederrors.NoConfigFileFoundError("Path to logging file does not exist!")

        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def get_input_configuration(self):
        """
        Get the input configuration

        Returns:
            self.input_configuration as a dict
        """
        if self.input_config is None:
            self.input_config = self.config.get_input_configuration()
        return self.input_config

    def set_input_configuration(self, input_config):
        """


        Args:
            input_config: Set the input config. Use a configuration from config classes

        Returns:
            None
        """
        self.input_config = input_config

    def create_input_interfaces(self):
        """
        Creates the one and only input interface
        Returns:
            None
        """
        self.input_config = self.config.get_input_configuration()
        self.input = if_factory.InterfaceFactory.produce(self.config.get_problem_type(),
                                                         self.input_config)

    def get_output_configurations(self):
        """
        Get the output configurations

        Returns:
            self.output_configurations as dicts
        """
        if len(self.output_configurations) == 0:
            self.output_configurations = self.config.get_output_configurations()
        return self.output_configurations

    def set_output_configurations(self, output_configurations):
        """
        Set the output configurations. Output configurations are instances of interface configurations

        Args:
            output_configurations:

        Returns:

        """
        self.output_configurations = output_configurations

    def create_output_interfaces(self):
        """
        Creates multiple output interfaces
        Returns:
            None
        """
        self.output = []
        for output_config in self.get_output_configurations():
            self.output.append(if_factory.InterfaceFactory.produce(
                self.config.get_problem_type(),
                output_config
            ))

    def set_raw_scenarios(self, raw_scenarios=None):
        """
        Here we create the raw scenarios ... we should implement raw_scenarios against an interface.

        Args:
            raw_scenarios: set the raw scenarios, which will be expand later

        Returns:
            None
        """
        if raw_scenarios is not None:
            self.raw_scenarios = raw_scenarios
        elif self.config.get_raw_scenarios_source() == 'file':
            self.raw_scenarios = self.config.get_scenarios()
        elif self.config.get_raw_scenarios_source() == 'database':
            self.raw_scenarios = self.input.get_raw_scenarios()
        else:
            errors.homebrewederrors.RawScenarioSourceNotKnown("Where do we get the raw scenarios from?!")

        self.logger.info("Rawscenarios: {}".format(self.raw_scenarios))

    def add_scenario(self, raw_scenario):
        """
        Add one raw scenario. This can be called from outside before we configure our scenarios and create our simulations.

        Args:
            raw_scenario: The raw scenario to be added

        Returns:
            None
        """
        self.scenarios.append(sc_factory.ScenariosFactory(
            problem_type=self.config.get_problem_type(),
            raw_scenario=raw_scenario,
            if_input=self.input,
            if_output=self.output
        ))

    def create_scenarios(self):
        """
        Creates a list of scenarios

        Returns:
            None
        """

        # FIXME self.input and self.output should be configurations and not an initialized object
        self.scenarios = sc_factory.ScenariosFactory.produce(
            problem_type=self.config.get_problem_type(),
            raw_scenarios=self.raw_scenarios,
            if_input=self.input,
            if_output=self.output)

    def store_scenarios(self):
        if self.config.get_store_scenarios_flag():
            self.input.store_scenarios(self.scenarios)

    def create_simulation_configurations(self):
        """
        After having created all concrete scenarios, we create all simulations from those scenario configurations.

        Returns:

        """

        for concrete_scenario in self.scenarios:
            concrete_scenario.configure_simulations(self.config.get_sim_strategy())

    def create_simulations(self):
        """
        After having created all simulation configurations, we create all simulation objects. This method should be called
        on the machine where the simulations are run.

        Returns:

        """

        for concrete_scenario in self.scenarios:
            concrete_scenario.create_simulations()

    def run_scenarios(self):
        for concrete_scenario in self.scenarios:
            if self.config.get_use_celery():
                concrete_scenario.run_celery()
            else:
                concrete_scenario.run()

    def start(self):
        """
        We are starting everything from here ...

        """
        # if len(sys.argv) == 1:
        #    # Create configuration reader later
        #    print "Run in default mode!"
        # else:
        #    self.cliparser.print_help()
        #    print "Run in command line mode!"

        # 1. Init config
        self.create_cli_parser()
        self.create_configuration()
        self.config.print_sections()

        # 2. Init logging
        self.create_logging()

        # 3. Init/output interfaces
        self.create_input_interfaces()
        self.create_output_interfaces()

        # 4. Init scenarios and the simulation configurations
        self.set_raw_scenarios()
        self.create_scenarios()
        self.store_scenarios()
        self.create_simulation_configurations()

        # 5. Run scenarios
        self.run_scenarios()
