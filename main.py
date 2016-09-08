# -*- coding: utf-8 -*-
"""
Main entry point of our framework.
"""

import control.control

__authors__ = "Johannes Zerwas, Andreas Blenk"

if __name__ == '__main__':
    ctrl = control.control.Control(configurationfile='control/configuration_default.ini')
    ctrl.start()
