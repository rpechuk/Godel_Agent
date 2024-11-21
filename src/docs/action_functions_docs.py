"""
This file defines various actions that can be used by an agent, such as displaying 
analysis, interacting with the environment, reading, modifying logic, executing code, and evaluating tasks.


Functions:


action_environment_aware:
Description: Reflect and summarize available resources of the current runtime environment including variables, functions, modules, and external libraries.
Returns: summary (str): Summary of available resources.


action_read_logic:
Description: Reads the source code of the specified logic (function, method, or class) within a given module.
    
Parameters:
module_name (str): The name of the module (e.g., 'agent_module').
target_name (str): The name of the function, method, or class (e.g., 'solver', 'Agent.action_call_llm', 'Agent').
    
Return Values:
code_str (str): A string representing the source code of the specified logic.



action_adjust_logic:
Description: Modify/Add/Delete the source code of the specified logic (function, method, or class) within a given module to 
improve task-solving ability or create a tool designed specifically to assist in task-solving efficiently.

Parameters:
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
    
    
action_run_code:
Description: Execute Python or shell code and capture the output, errors, and return value. 
(Running python code can get and store objects designed specifically to assist in task-solving efficiently, such as prompts)
    
Parameters:
code_type (str): The type of code to execute ('python' or 'bash').
code (str): The code to execute as a string.
timeout (float): Maximum execution time in seconds (default: 30.0).
    
Returns:
result_str (str): A string summarizing the output, errors, and return value.


safe_eval:
Description: Safely evaluate an expression.

Parameters:
expr (str): The expression to evaluate.
globals_dict (dict): The global scope for evaluation.
locals_dict (dict): The local scope for evaluation.
Return Value:
The result of the evaluated expression, or None if the evaluation fails.

action_evaluate_on_task:
Description: Evaluate the current solver on the goal task samples and return the evaluation feedback.
Parameters:
task: The task instance that contains the evaluation logic.
solver: The solver or model to evaluate against the task.

Return Values:
feedback (str): Evaluation feedback including valid set accuracy, test set accuray, test sample inputs, model outputs and valid sample answer.
"""