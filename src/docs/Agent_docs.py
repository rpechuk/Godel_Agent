"""
This file defines the main Agent class, which orchestrates interactions with an LLM, 
executes tool-based actions, and iteratively optimizes its problem-solving capabilities.
The following functions are defined for the agent class:

__init__
Description: Initializes the agent with a goal prompt, a task configuration, an LLM client, and internal states for optimization history and action tracking.
Parameters:
agent (Agent): The OpenAI client instance used to interact with the LLM.
goal_prompt_path (str): Path to the goal prompt markdown file. Defaults to 'goal_prompt.md'.
Return Values: None.


reinit
Description: Reinitializes the agent’s optimization history and logs the current environment state and solver logic.
Parameters: agent (Agent): The OpenAI client instance used to interact with the LLM.
Return Values: None.


execute_action
Description: Executes the function called by the model and returns the result.
Parameters:
agent (Agent): The OpenAI client instance used to interact with the LLM.
actions (Dict): A dictionary containing tool call details and arguments.
Return Values: None.


evolve
Description: Evolves the agent by prompting the LLM to suggest improvements.
Parameters: agent (Agent): The OpenAI client instance used to interact with the LLM..
Return Values: None.


action_call_json_format_llm
Description: Sends a prompt to the LLM, requesting responses in JSON format based on specified keys.
Parameters:
agent (Agent): The OpenAI client instance used to interact with the LLM.
messages (List[Dict]): A list of message dictionaries forming the conversation context.
model (str): Specifies the LLM model to use. Defaults to "gpt-4o-mini".
temperature (float): Controls randomness in responses. Defaults to 1.0.
max_completion_tokens (int): Maximum token count for the response. Defaults to 4096.
num_of_response (int): Number of responses to generate. Defaults to 1.
role (str): Specifies the role of the LLM. Defaults to "task solver".
return_dict_keys (List[str]): Keys required in the JSON response.
requirements (str): Additional instructions for the LLM.
Return Values: List[Dict]: JSON responses with required keys.



action_call_llm
Description: Sends a request to the OpenAI LLM with a system prompt and user message, and returns the response.

Parameters:
agent (Agent): The OpenAI client instance used to interact with the LLM.
messages (List[Dict[str, str]]): A list of message dictionaries (conversation history).
response_format (str): The desired format of the LLM's output.
model (str): Specifies which LLM model to use.
temperature (float): A float value controlling the randomness of the model's responses. Higher values (e.g., 1.0) increase creativity, while lower values (e.g., 0.1) make the responses more focused and deterministic.
max_completion_tokens: An integer defining the maximum number of tokens in the completion response, up to 4096.
n (int): The number of chat completion choices to generate for each input message.

Return Values:
    response (dict): The response from the OpenAI LLM.
    
"""