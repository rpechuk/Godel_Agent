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
import tasks.task_drop as task_drop

def solver(agent, task: str):
    messages = [{"role": "user", "content": f"# Your Task:\n{task}"}]
    response = agent.action_call_json_format_llm(
        model="gpt-3.5-turbo", 
        messages=messages, 
        temperature=0.5, 
        num_of_response=1,
        role="read comprehension expert", 
        return_dict_keys=["reasoning", "answer"], 
        requirements=(
            "1. Please explain step by step.\n"
            "2. Directly answer the question.\n"
            "3. The answer MUST be a concise string.\n"
        ).strip(), 
    )
    
    return_dict = response[0]
    return_dict["answer"] = str(return_dict.get("answer", ""))
    return return_dict