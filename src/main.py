# from agent_module import Agent, self_evolving_agent

from agent_modules import *
import agent_modules.Agent as Agent

if __name__ == "__main__":
    for _ in range(1):
        self_evolving_agent = Agent.Agent(goal_prompt_path="goal_prompt2.md")
        self_evolving_agent.reinit()
        self_evolving_agent.evolve()