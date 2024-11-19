from agent_module import Agent, self_evolving_agent

if __name__ == "__main__":
    for _ in range(1):
        self_evolving_agent = Agent(goal_prompt_path="goal_prompt.md")
        self_evolving_agent.reinit()
        self_evolving_agent.evolve()