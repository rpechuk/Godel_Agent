tools = [{
                "type": "function",
                "function": {
                    "name": "action_display_analysis",
                    "description": "Display an analysis of the current state, including available resources, logic (of solver or other actions) and evaluation feedbacks from the target task, and reasons or plans for the next actions based on this analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis": {
                                "type": "string",
                                "description": "A detailed analysis of the current state, including reasons or plans for the following actions."
                            }
                        },
                        "required": ["analysis"],
                        "additionalProperties": False,
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_environment_aware",
                    "description": "Reflect and summarize available resources of the current runtime environment including variables, functions, modules, and external libraries.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_read_logic",
                    "description": "Reads the source code of the specified logic (function, method, or class) within a given module.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "The module where the logic resides."
                            },
                            "target_name": {
                                "type": "string",
                                "description": "The name of the function, method, or class to read. If the target_name contains a dot, it refers to a method within a class (e.g., 'Agent.action_call_llm')."
                            }
                        },
                        "required": ["module_name", "target_name"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_adjust_logic",
                    "description": "Modify/Add/Delete the source code of the specified logic (function, method, or class) within a given module to improve task-solving ability or create a tool designed specifically to assist in task-solving efficiently.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "The module where the logic resides."
                            },
                            "target_name": {
                                "type": "string",
                                "description": "The name of the function, method, or class to modify/add/delete. If the target_name contains a dot, it refers to a method within a class."
                            },
                            "new_code": {
                                "type": "string",
                                "description": "The new logic as a string. (Ensure there is no extra indentation in new_code)"
                            },
                            "target_type": {
                                "type": "string",
                                "enum": ["function", "class"],
                                "description": "The type of target."
                            },
                            "operation": {
                                "type": "string",
                                "enum": ["modify", "add", "delete"],
                                "description": "The operation to perform."
                            }
                        },
                        "required": ["module_name", "target_name", "new_code", "target_type", "operation"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_run_code",
                    "description": "Execute Python or shell code and capture the output, errors, and return value. (Running python code can get and store objects designed specifically to assist in task-solving efficiently, such as prompts)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code_type": {
                                "type": "string",
                                "enum": ["python", "bash"],
                                "description": "The type of code to execute."
                            },
                            "code": {
                                "type": "string",
                                "description": "The code to execute as a string."
                            }
                        },
                        "required": ["code_type", "code"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_call_json_format_llm",
                    "description": "Call an external LLM for assistance with gathering insights, refining strategies, correcting errors, and solving complex problems. Output response in JSON format.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "enum": ["gpt-4o-mini", "gpt-4o"],
                                "description": "ID of the model to use."
                            },
                            "messages": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {
                                            "enum": ["system", "assistant", "user"]
                                        },
                                        "content": {"type": "string"}
                                    },
                                    "required": ["role", "content"],
                                    "additionalProperties": False
                                },
                                "description": "A list of messages comprising the conversation so far."
                            },
                            "temperature": {
                                "type": "number",
                                "description": "What sampling temperature to use. Higher values will make the output more random, while lower values will make it more focused and deterministic."
                            },
                            "role": {
                                "type": "string",
                                "description": "The role that LLM play."
                            },
                            "return_dict_keys": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "An array containing the names of the keys that should be present in the returned dictionary."
                            },
                            "requirements": {
                                "type": "string",
                                "description": "A string that specifies the conditions required to perform a call to the LLM."
                            }
                        },
                        "required": ["model", "messages", "temperature", "role", "return_dict_keys", "requirements"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "action_evaluate_on_task",
                    "description": "Evaluate the current solver on the goal task samples and return the evaluation feedback including valid set accuracy, test set accuray, test sample inputs, model outputs and valid sample answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]