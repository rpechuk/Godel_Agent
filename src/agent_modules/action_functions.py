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



# action_counter = collections.defaultdict(int)

def action_display_analysis(analysis):
    print(analysis, "\n\n")
    return "Analysis Received. Just do it!"


def action_environment_aware(agent: AgentBase):
    """
    Reflect and summarize available resources of the current runtime environment including variables, functions, modules, and external libraries.

    Returns:
        summary (str): Summary of available resources.
    """
    def summarize_items(items, header):
        summary = [header]
        for name, value in items:
            if not name.startswith('__'):
                if name in ['goal_prompt']:
                    summary.append(f"- {name} = Your {name}.")
                elif name in ['optimize_history', 'function_map', 'action_functions']:
                    summary.append(f"- {name} = The length of your {name} is {len(getattr(agent, name))}.")
                elif name in ['logic']:
                    pass
                else:
                    summary.append(f"- {name} = {value}")
        if len(summary) == 1:
            summary.append("- None")
        return summary
    
    summary = []
    
    global_vars = [(k, v) for k, v in globals().items() if not k.startswith('__') and k != "AgentBase"]
    functions = [(k, v) for k, v in global_vars if inspect.isfunction(v)]
    calsses = [(k, v) for k, v in global_vars if inspect.isclass(v)]
    modules = [(k, v) for k, v in global_vars if inspect.ismodule(v)]
    variables = [(k, v) for k, v in global_vars if not (inspect.isfunction(v) or inspect.isclass(v) or inspect.ismodule(v))]
    
    summary.extend(summarize_items(functions, "\nGlobal Functions:"))
    summary.extend(summarize_items(modules, "\nGlobal Modules:"))
    summary.extend(summarize_items(variables, "\nGlobal Variables:"))
    summary.extend(summarize_items(calsses, "\nGlobal Calsses:"))
    
    methods = inspect.getmembers(agent, inspect.ismethod)
    attributes = inspect.getmembers(agent, lambda x: not inspect.ismethod(x))
    
    summary.extend(summarize_items(methods, "\nCurrent Agent Instance's Methods:"))
    summary.extend(summarize_items(attributes, "\nCurrent Agent Instance's Attributes:"))

    return "\n".join(summary).strip()

def action_read_logic(module_name: str, target_name: str):
    """
    Reads the source code of the specified logic (function, method, or class) within a given module.
    
    Args:
        module_name (str): The name of the module (e.g., 'agent_module').
        target_name (str): The name of the function, method, or class (e.g., 'solver', 'Agent.action_call_llm', 'Agent').
    
    Returns:
        code_str (str): A string representing the source code of the specified logic.
    """
    try:
        # Dynamically import the module
        module = importlib.import_module(module_name)
        
        # If target_name contains a dot, it's a method in a class (e.g., 'Agent.evolve')
        if '.' in target_name:
            class_name, target_name = target_name.split('.')
            target_class = getattr(module, class_name)
            target = getattr(target_class, target_name)
        else:
            # Otherwise, it's a top-level function or class
            target = getattr(module, target_name)
        
        # Extract the source code using inspect
        code_str = logic.get_source_code(target, target_name)
        
        return code_str
    
    except Exception as e:
        raise e
    
def action_adjust_logic(module_name: str, target_name: str, new_code=str, target_type: str = 'function', operation: str = 'modify'):
    """
    Modify/Add/Delete the source code of the specified logic (function, method, or class) within a given module to 
    improve task-solving ability or create a tool designed specifically to assist in task-solving efficiently.

    Args:
        module_name (str): The name of the module to modify (e.g., 'agent_module').
        target_name (str): The name of the function, method, or class to do operation (e.g., 'solver').
        new_code (str): The new logic as a string (including `def` for functions or `class` or classes). For delete, it can be empty string.
        target_type (str): The type of target ('function', 'class'). Default is 'function'.
        operation (str): The type of operation to perform ('modify', 'add', or 'delete'). Default is 'modify'.

    Raises:
        ValueError: Unknown operation

    Examples:
        >>> modify_logic('agent_module', 'evolve', 'def evolve(agent):\\n    print("New evolve method")', target_type='function')
        >>> modify_logic('agent_module', 'evolve', '', target_type='function', operation='delete')
    """
    if module_name == "agent_module":
        if target_name == "solver":
            if "gpt-4o" in new_code:
                raise ValueError("ONLY model **gpt-3.5-turbo** can be used in solver.")
            if "time.sleep" in new_code:
                raise ValueError("Don't use `time.sleep` in solver.")
        if target_name == "Agent.action_call_llm":
            raise ValueError("Don't modify `action_call_llm`.")
        if target_name == "Agent.action_call_json_format_llm":
            raise ValueError("Don't modify `action_call_json_format_llm`.")

    if "import logging" in new_code or "from logging" in new_code:
        raise ValueError("Don't use `logging`.")

    # Import the module dynamically
    module = importlib.import_module(module_name)
    _target_name = target_name
    print(new_code, end='\n\n')
    # Perform the operation based on type (modify, add, delete)
    if operation in ['modify', 'add']:
        # Compile the new code within the current global and a new local dict
        locals_dict = {}
        exec(compile(new_code, f"running.{module_name}.{target_name}", "exec"), globals(), locals_dict)
        if '.' in target_name:
            class_name, target_name = target_name.split('.')
            if class_name in locals_dict:
                new_target = getattr(locals_dict[class_name], target_name)
                locals_dict.pop(class_name)
            else:
                new_target = locals_dict[target_name]
                locals_dict.pop(target_name)
        else:
            new_target = locals_dict[target_name]
            locals_dict.pop(target_name)
        globals().update(locals_dict)
        
        # Apply the new definition or value to the target
        if '.' in target_name:  # Class attribute
            class_name, target_name = target_name.split('.')
            cls = getattr(module, class_name)
            setattr(cls, target_name, new_target)
            getattr(cls, target_name).__source__ = new_code
            # Add or update the __source__ attribute on the class level to store the full new definition
            if not hasattr(cls, '__source__'):
                cls.__source__ = {}
                for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                    cls.__source__[name] = logic.get_source_code(method, name)
            cls.__source__[target_name] = '\n'.join(['    '+code_line for code_line in new_code.split('\n')])

        else:  # Module level attribute
            setattr(module, target_name, new_target)
            getattr(module, target_name).__source__ = new_code

    elif operation == 'delete':
        if '.' in target_name:  # Class attribute
            class_name, target_name = target_name.split('.')
            cls = getattr(module, class_name)
            delattr(cls, target_name)
            if hasattr(cls, '__source__') and target_name in cls.__source__:
                del cls.__source__[target_name]
        else:  # Module level attribute
            delattr(module, target_name)

    else:
        raise ValueError(f"Unknown operation '{operation}'. Expected 'modify', 'add', or 'delete'.")

    return f"Successfully {operation} `{module_name}.{_target_name}`."

def action_run_code(code_type: str, code: str, timeout: float = 30.0) -> str:
    """
    Execute Python or shell code and capture the output, errors, and return value. 
    (Running python code can get and store objects designed specifically to assist in task-solving efficiently, such as prompts)
    
    Args:
        code_type (str): The type of code to execute ('python' or 'bash').
        code (str): The code to execute as a string.
        timeout (float): Maximum execution time in seconds (default: 30.0).
    
    Returns:
        result_str (str): A string summarizing the output, errors, and return value.
    """
    
    def safe_eval(expr: str, globals_dict, locals_dict):
        """Safely evaluate an expression."""
        try:
            tree = ast.parse(expr, mode='eval')
            return eval(compile(tree, '<string>', 'eval'), globals_dict, locals_dict)
        except Exception:
            return None

    if code_type.lower() == 'python':
        output = io.StringIO()
        error_output = io.StringIO()
        return_value = None
        
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):                
            locals_dict = {}
            exec(code, globals(), locals_dict)
            globals().update(locals_dict)
            
            # Safely evaluate the last expression
            return_value = safe_eval(code.splitlines()[-1], globals(), locals_dict)
        
        result = {
            "output": output.getvalue(),
            "errors": error_output.getvalue(),
            "return_value": return_value
        }
        
        output.close()
        error_output.close()
    
    elif code_type.lower() == 'bash':
        try:
            process = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            result = {
                "output": process.stdout,
                "errors": process.stderr,
                "return_value": process.returncode
            }
        except subprocess.TimeoutExpired:
            result = {
                "output": "",
                "errors": f"Command timed out after {timeout} seconds",
                "return_value": None
            }
        except Exception as e:
            result = {
                "output": "",
                "errors": repr(e),
                "return_value": None
            }
            
    else:
        return "Error: Unsupported code_type. Only 'python' and 'bash' are supported."

    # Format the result
    result_str = f"Execution Summary ({code_type.capitalize()}):\n"
    if result["output"]:
        result_str += f"Output:\n{result['output']}\n"
    if result["errors"]:
        result_str += f"Errors:\n{result['errors']}\n"
    if result["return_value"] is not None:
        result_str += f"Return Value: {result['return_value']}\n"
    
    return result_str or "No output, errors, or return value."



def action_evaluate_on_task(task, solver):
    """
    Evaluate the current solver on the goal task samples and return the evaluation feedback.

    Returns:
        feedback (str): Evaluation feedback including valid set accuracy, test set accuray, test sample inputs, model outputs and valid sample answer.
    """
    feedback, acc = task.evaluate(solver)
    if acc > tasks.task_mgsm.last_test_acc:
        logic.store_all_logic(f"../{tasks.task_mgsm.__name__}_{round(acc, 4)}")
        tasks.task_mgsm.last_test_acc = acc
    return feedback