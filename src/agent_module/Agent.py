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

from rag import StructuredRAG
import agent_module.AgentBase as AgentBase
import agent_module.action_functions as action_functions
import agent_module.solver as solver
import agent_module.action_tools as action_tools
import tasks.task_drop


tool_call_examples = [{
    'content': '',
    'role': 'assistant',
    'tool_calls': [{
        'id': 'call_9xqxybrj',
        'function': {'arguments': '{"module_name":"agent_module","target_name":"solver"}','name': 'action_read_logic'},
        'type': 'function'
    }]
}, {
    'role': 'tool',
    'content': action_functions.action_read_logic("agent_module", "solver"),
    'tool_call_id': 'call_9xqxybrj'
}, {
    'content': '',
    'role': 'assistant',
    'tool_calls': [{
        'id': 'call_ac28pj9s',
        'function': {'arguments': '{"analysis":"The current solver logic reads the task, sends it to an LLM for processing with specific parameters (model, messages, temperature, number of responses, role, and requirements). The response is expected to include "reasoning" and "answer" keys. The answer is then converted to a string before being returned."}','name': 'action_display_analysis'},
        'type': 'function'
    }]
}, {
    'role': 'tool',
    'content': action_functions.action_display_analysis("The current solver logic reads the task, sends it to an LLM for processing with specific parameters (model, messages, temperature, number of responses, role, and requirements). The response is expected to include 'reasoning' and 'answer' keys. The answer is then converted to a string before being returned."),
    'tool_call_id': 'call_ac28pj9s'
}, {
    'content': '',
    'role': 'assistant',
    'tool_calls': [{
        'id': 'call_pm8mijoa',
        'function': {'arguments': '{"code_type":"python","code":"def action_print_hello():\n    print("Hello World!")\naction_print_hello()"}','name': 'action_run_code'},
        'type': 'function'
    }]
}, {
    'role': 'tool',
    'content': action_functions.action_run_code("python", "def action_print_hello():\n    print('Hello World!')\naction_print_hello()"),
    'tool_call_id': 'call_pm8mijoa'
}, {
    'content': '',
    'role': 'assistant',
    'tool_calls': [{
        'id': 'call_c3m2908m',
        'function': {'arguments': '{"module_name":"agent_module","target_name":"action_print_hello","new_code":"def action_print_hello():\n    print("Hello World!"")","operation","add"}','name': 'action_adjust_logic'},
        'type': 'function'
    }]
}, {
    'role': 'tool',
    'content': action_functions.action_adjust_logic("agent_module", "action_print_hello", "def action_print_hello():\n    print('Hello World!')", operation="add"),
    'tool_call_id': 'call_c3m2908m'
}, {
    'content': '',
    'role': 'assistant',
    'tool_calls': [{
        'id': 'call_pm8mijoa',
        'function': {'arguments': '{"code_type":"bash","code":"ls -l"}','name': 'action_run_code'},
        'type': 'function'
    }]
}, {
    'role': 'tool',
    'content': action_functions.action_run_code("bash", "ls -l"),
    'tool_call_id': 'call_pm8mijoa'
}]

class Agent(AgentBase.AgentBase):
    def __init__(agent, goal_prompt_path='goal_prompt.md'):
        # Load configurations
        agent.goal_prompt = open(goal_prompt_path, 'r').read()
        agent.goal_task = tasks.task_drop.DROP_Task()
        agent.client = openai.OpenAI(api_key='ollama', base_url="http://localhost:11434/v1")

        # Initialize optimization history and iterations

        agent.action_functions = action_tools.tools
        
        agent.action_counter = collections.defaultdict(int)

        agent.optimize_history = []

        # Initialize RAG system
        agent.rag = StructuredRAG()

        #  Add few shot examples of tool calls
        agent.optimize_history.extend(tool_call_examples)
        
        
    def reinit(agent):
        agent.optimize_history = []
        first_aware_content = action_functions.action_environment_aware(agent)
        solver_logic = action_functions.action_read_logic("agent_module", "solver")
        
        print(first_aware_content, end="\n\n")
        print(solver_logic, end="\n\n")

        # agent.optimize_history.append({"role": "user", "content": first_aware_content})
        agent.optimize_history.append({"role": "user", "content": "The logic of solver:\n" + solver_logic})
        agent.optimize_history.extend(tool_call_examples)

    def execute_action(agent, actions: typing.Dict):
        """
        Executes the function called by the model and returns the result.
        """
        
        if 'tool_calls' not in actions:
            print("Agent Evolve - NO TOOLS CALLED :(", end="\n\n")
            agent.optimize_history.append({
                "role": "user",
                "content": """Don't forget the following things:
1. You must respond with at least one tool call. Things to remember about calling the tools you are provided:
    - You are provided with function signatures within <tools></tools> XML tags
    - For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags
2. Your task is to improve the solver function's performance on the mgsm benchmark task.
    - An example input is: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
    - The correct output for the above example is 3
3. If you are ready to run the task, then call `action_evaluate_on_task` with the relevant arguments."""
})
            agent.evolve()
            return

        is_reinit = False
        for tool_call in actions['tool_calls']:
            print("tool call:", tool_call, end="\n\n")
            try:
                agent.action_counter[tool_call['function']['name']] += 1
                arguments = json.loads(tool_call['function']['arguments']) if tool_call['function']['arguments'] else {}
                if tool_call['function']['name'] == "action_display_analysis":
                    result = action_functions.action_display_analysis(**arguments)

                elif tool_call['function']['name'] == "action_environment_aware":
                    result = action_functions.action_environment_aware(agent, **arguments)

                elif tool_call['function']['name'] == "action_read_logic":
                    result = action_functions.action_read_logic(**arguments)

                elif tool_call['function']['name'] == "action_adjust_logic":
                    result = action_functions.action_adjust_logic(**arguments)

                elif tool_call['function']['name'] == "action_run_code":
                    result = action_functions.action_run_code(**arguments)
                    if arguments.get("code_type", None) == "python" and "self_evolving_agent.reinit()" in arguments.get("code", ""):
                        is_reinit = True
                elif tool_call['function']['name'] == "action_call_llm":
                    result = agent.action_call_llm(**arguments)
                    print(result[0])

                elif tool_call['function']['name'] == 'action_call_json_format_llm':
                    result = agent.action_call_json_format_llm(**arguments)
                    try:
                        print(json.loads(result[0]))
                    except:
                        print(result[0])

                elif tool_call['function']['name'] == "action_evaluate_on_task":
                    result = action_functions.action_evaluate_on_task(agent.goal_task, functools.partial(solver.solver, agent))
                else:
                    raise ValueError(f"Unknown function name: {tool_call['function']['name']}")

            except Exception as e:
                agent.action_counter["error_handle"] += 1
                exception_stringio = io.StringIO()
                traceback.print_exc(file=exception_stringio)
                result = "Error " + exception_stringio.getvalue()
                exception_stringio.close()

            print("tool call result:\n", result, sep="", end="\n\n")
            if is_reinit:
                break
            agent.optimize_history.append({"role": "tool", 
                                           "content": result, 
                                            "tool_call_id": tool_call['id']})


        print("Action Counter:", agent.action_counter, end='\n\n')
        if agent.action_counter["evolve"] >= 30:
            sys.exit(1)
        print("Agent Evolve", end="\n\n")
        
        agent.evolve()

    def evolve(agent):
        """
        Evolves the agent by prompting the LLM to suggest improvements.
        """
        print('-' * 120)
        agent.action_counter["evolve"] += 1

        tool_call_ids = set()
        remain_optimize_history = []
        for message in agent.optimize_history[-10:]:
            if message["role"] == "assistant" and 'tool_calls' in message:
                tool_call_ids = set()
                for tool_call in message["tool_calls"]:
                    tool_call_ids.add(tool_call["id"])
            if message["role"] == "tool" and message["tool_call_id"] not in tool_call_ids:
                print(f"pop item: {message}", end='\n\n')
                continue
            remain_optimize_history.append(message)
        agent.optimize_history = remain_optimize_history

        messages = [{"role": "system", "name": "Principles", "content": agent.goal_prompt}, 
                    {"role": "system", "name": "Environment", "content": action_functions.action_environment_aware(agent)},
                    *agent.optimize_history]
        
        # NOTE(Ron): This doesn't work but i'm leaving it here for testing sake and also to show how I debugged the issue
        # if action_counter["evolve"] == 1:
        #     result = action_evaluate_on_task(agent.goal_task, functools.partial(solver, agent))
        #     with open("result.txt", "w") as f:
        #         f.write(result)
        #     messages.append({"role": "user", "content": "The intial evaluation result was:\n" + str(result)})

        try:
            response = agent.action_call_llm(messages=messages, model="gpt-4o", response_format="text", tools=agent.action_functions, context=True)
            print(response)
        except Exception as e:
            print(repr(e))
            for message in messages:
                print(message)
            sys.exit(1)
        
        agent.optimize_history.append(response[0])
        agent.execute_action(response[0])

    def action_call_json_format_llm(
        agent,
        *,
        messages: typing.List[typing.Dict[str, str]], 
        model: typing.Literal["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"] = "gpt-4o-mini", 
        temperature: float = 1.0, 
        max_completion_tokens: int = 4096, 
        num_of_response: int = 1,
        role: str = "task solver", 
        return_dict_keys: typing.List[str] = [], 
        requirements: str = "", 
    ):
        system_prompt = (
            f"You are a helpful {role}.\n"
            f"Reply in JSON format, ONLY using the keys {return_dict_keys}.\n"
            f"Requirements:\n{requirements}"
        ).strip()
        _messages = [{"role": "system", "content": system_prompt}, *messages]
        return_dicts = agent.action_call_llm(model=model,
                                    messages=_messages, 
                                    temperature=temperature,
                                    max_completion_tokens=max_completion_tokens,
                                    n=num_of_response,
                                    response_format="json")
        
        for key in return_dict_keys:
            for return_dict in return_dicts:
                if key not in return_dict:
                    return_dict[key] = f"NO {key} IN DICTIONARY"
        return return_dicts
    
    def action_call_llm(
        agent, 
        *,
        model: typing.Literal["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"] = "gpt-4o-mini", 
        messages: typing.List[typing.Dict[str, str]], 
        temperature: float = 1.0, 
        max_completion_tokens: int = 4096, 
        n: int = 1,
        response_format: typing.Literal["text", "json", "json_object"] = "text", 
        tools=None, 
        context=False
    ):
        """
        Sends a request to the OpenAI LLM with a system prompt and user message, and returns the response.

        Args:
            agent (Agent): The OpenAI client instance used to interact with the LLM.
            messages (List[Dict[str, str]]): A list of message dictionaries (conversation history).
            response_format (str): The desired format of the LLM's output.
            model (str): Specifies which LLM model to use.
            temperature (float): A float value controlling the randomness of the model's responses. Higher values (e.g., 1.0) increase creativity, while lower values (e.g., 0.1) make the responses more focused and deterministic.
            max_completion_tokens: An integer defining the maximum number of tokens in the completion response, up to 4096.
            n (int): The number of chat completion choices to generate for each input message.

        Returns:
            response (dict): The response from the OpenAI LLM.
        """
        try:
            if response_format == "json":
                response_format = "json_object"
            
            import copy
            messages = copy.deepcopy(messages)
            for message in messages:
                message["content"] = str(message["content"])
            
            kwargs = {
                "n": n,
                "model": model,
                "messages": messages,
                "response_format": {"type": response_format if response_format == "json_object" else "text"}, 
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens
            }

            if tools is not None:
                kwargs["tools"] = tools
                # kwargs["tool_choice"] = tool_choice # NOTE: this does nothing with Ollama right now

            if context:
                ctx = agent.rag.get_context(messages[-1]["content"], k=1)
                messages.insert(2, {"role": "system",  "name": "context", "content": "**Use the following context about the system to inform your next tool calls:**\n" + ctx})

                # print("Messages:")
                # pprint.pp(messages[2:])
                print("\n\n")

            response = agent.client.chat.completions.create(**kwargs).to_dict() # to Python dictionary
            
            def try_parse_json(content):
                try:
                    return json.loads(content)
                except:
                    return {"JSONDecodeError": content}

            if response_format == "text":
                return [choice["message"] for choice in response["choices"]]
            else:
                return [try_parse_json(choice["message"]["content"]) for choice in response["choices"]]
        except Exception as e:
            raise e

# self_evolving_agent = Agent()