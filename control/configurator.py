# -*- coding: utf-8 -*-
# TODO add description

import ConfigParser
import input_output.data_interface.interface_config as if_config
import os.path
import errors.homebrewederrors
import logging

__authors__ = "Patrick Kalmbach, Johannes Zerwas, Andreas Blenk, Michael Manhart"


class Config(object):
    """
    Holds the configuration provided by the configuration file and makes them easily accessible
    Attributes:
        config: configuration parser
    """

    def __init__(self, configfile, cli_args=None):
        # Check if a configfile is specified in the cli, if it is replace the configfile with the cli configfile
        if cli_args is not None:
            if cli_args.configfile is not None:
                configfile = cli_args.configfile
        if configfile is None:
            raise errors.homebrewederrors.NoConfigFileFoundError('No configuration file specified')
        self.config = ConfigParser.SafeConfigParser()
        self.cli_args = cli_args
        # TODO: Use the other cli values to overwrite the data in the config file
        if os.path.isfile(configfile):
            self.config.read(configfile)
        else:
            raise errors.homebrewederrors.NoConfigFileFoundError("Could not find configfile. Check path!")
        self.logger = logging.getLogger('Configurator-Logger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Finished __init__ config!")

    def get_raw_scenarios_source(self):
        return self.config.get('Main', 'raw_scenarios_source')

    def get_store_scenarios_flag(self):
        return self.config.getboolean('Main', 'store_scenarios_flag')

    def get_problem_type(self):
        return self.config.get('Main', 'problem_type')

    def get_use_celery(self):
        return self.config.getboolean('Main', 'celery')

    def get_sim_strategy(self):
        return self.config.get('Main', 'sim_strategy')

    def get_celery_config(self):
        config_as_dict = {}
        for name, value in self.config.items('Celery'):
            config_as_dict[name] = value
        return config_as_dict

    def _get_interface_configuration_for_name(self, name=''):
        """
        Returns the configuration of the interface with the provided name
        Args:
            name: Name of the interface

        Returns:
            the InterfaceConfig containing the values read from the configuration file
        """
        if name == '':
            self.logger.warning('argument name is not set')
        config_as_dict = {}
        for name, value in self.config.items(name):
            config_as_dict[name] = value
        return config_as_dict

    def get_input(self):
        return self.config.get('Main', 'input')

    def get_input_configuration(self):
        """
        Returns the configuration of the input interface
        Returns:
            InterfaceConfig containing the configuration of the input interface
        """
        config_as_dict = self._get_interface_configuration_for_name(self.get_input())
        return if_config.InterfaceConfigFactory.produce(config_as_dict)

    def get_output(self):
        return self.config.get('Main', 'output')

    def get_output_configurations(self):
        """
        Returns the configurations of the output interfaces
        Returns:
            List of InterfaceConfig containing the configuration of all output interfaces that should be used
        """
        outputs = self.get_output().split(',')
        configs = []
        for output in outputs:
            config_as_dict = self._get_interface_configuration_for_name(output.strip())
            configs.append(if_config.InterfaceConfigFactory.produce(config_as_dict))
        return configs

    def get_input_set_id(self):
        return self.config.getint('Main', 'input_set_id')

    def get_setup_id(self):
        return self.config.getint('Main', 'setup_id')

    def get_scenario_name(self):
        return self.config.get('Main', 'scenario')

    def get_scenarios(self):
        """
        Returns the parameters of the scenarios that should be created
        Returns:
            Parameters of the scenarios as a dict. The values of the dict are lists
        """
        list_with_tuples = self.config.items(self.get_scenario_name())
        config_as_dict = {}
        for name, value in list_with_tuples:
            config_as_dict[name] = value.split(',')
            if name == 'num_runs':
                if len(config_as_dict[name]) != 1:
                    raise ValueError('Setting for %s must be 1 Integer' % (name))
                else:
                    config_as_dict[name] = config_as_dict[name][0]

        return config_as_dict

    # Logging
    def get_logging_default_path(self):
        return self.config.get('Logging', 'default_path')

    def get_logging_default_level(self):
        return self.config.get('Logging', 'default_level')

    def get_logging_env_key(self):
        return self.config.get('Logging', 'env_key')

    def print_sections(self):
        for each_section in self.config.sections():
            print "------ Section: {} ------".format(each_section)
            for (each_key, each_val) in self.config.items(each_section):
                print "Config key: {} Value: {}".format(each_key, each_val)
            print "\n"


class ConfigError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)
