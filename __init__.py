"""
J.A.R.V.I.S. - Advanced Personal Assistant System
Ultra-advanced AI assistant for Windows PC control and automation
"""

__version__ = "2.0.0"
__author__ = "Supernova Corp"
__description__ = "Advanced AI Personal Assistant for Windows"

import sys
import os

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

__all__ = [
    'JARVIS',
    'SystemMonitor',
    'VoiceInterface',
    'ApplicationController',
    'FileManager',
    'NetworkManager',
    'SecurityManager',
    'PluginManager'
]