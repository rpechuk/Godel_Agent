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
import agent_modules.actions as actions
import agent_modules.solver as solver
import agent_modules.action_tools as action_tools


class Agent(AgentBase.AgentBase):
    def __init__(agent, goal_prompt_path='goal_prompt.md'):
        # Load configurations
        agent.goal_prompt = open(goal_prompt_path, 'r').read()
        agent.goal_task = tasks.task_mgsm.MGSM_Task()
        agent.client = openai.OpenAI(api_key='ollama', base_url="http://localhost:11434/v1")

        # Initialize optimization history and iterations

        agent.action_functions = action_tools.tools #[

        agent.optimize_history = []
        
        agent.action_counter = collections.defaultdict(int)

    def reinit(agent):
        agent.optimize_history = []
        first_aware_content = actions.action_environment_aware(agent)
        solver_logic = actions.action_read_logic("agent_module", "solver")
        
        print(first_aware_content, end="\n\n")
        print(solver_logic, end="\n\n")

        # agent.optimize_history.append({"role": "user", "content": first_aware_content})
        agent.optimize_history.append({"role": "user", "content": "The logic of solver:\n" + solver_logic})

    def execute_action(agent, actions: typing.Dict):
        """
        Executes the function called by the model and returns the result.
        """
        
        if 'tool_calls' not in actions:
            print("Agent Evolve - NO TOOLS CALLED :(", end="\n\n")
            agent.optimize_history.append({"role": "user", "content": "Don't forget that you must have at least one tool call in your actions."})
            agent.evolve()
            return

        is_reinit = False
        for tool_call in actions['tool_calls']:
            print("tool call:", tool_call, end="\n\n")
            try:
                agent.action_counter[tool_call['function']['name']] += 1
                arguments = json.loads(tool_call['function']['arguments']) if tool_call['function']['arguments'] else {}
                if tool_call['function']['name'] == "action_display_analysis":
                    result = actions.action_display_analysis(**arguments)

                elif tool_call['function']['name'] == "action_environment_aware":
                    result = actions.action_environment_aware(agent, **arguments)

                elif tool_call['function']['name'] == "action_read_logic":
                    result = actions.action_read_logic(**arguments)

                elif tool_call['function']['name'] == "action_adjust_logic":
                    result = actions.action_adjust_logic(**arguments)

                elif tool_call['function']['name'] == "action_run_code":
                    result = actions.action_run_code(**arguments)
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
                    result = actions.action_evaluate_on_task(agent.goal_task, functools.partial(solver, agent))
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
                    {"role": "system", "name": "Environment", "content": actions.action_environment_aware(agent)},
                    *agent.optimize_history]

        print("Messages:")
        pprint.pp(messages[2:])
        print("\n\n")

        try:
            response = agent.action_call_llm(messages=messages, model="gpt-4o", response_format="text", tools=agent.action_functions, tool_choice="required")
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
        tool_choice=None,
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
