import os
import io
import re
import ast
import sys
import json
import typing
import inspect
import functools
import itertools
import traceback
import importlib
import subprocess
import contextlib
import collections
import openai
import logic
import pprint
import tasks
import agent_modules.AgentBase as AgentBase
import agent_modules.Agent as Agent
import agent_modules.solver as solver

self_evolving_agent = Agent()


