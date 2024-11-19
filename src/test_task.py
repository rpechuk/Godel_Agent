from agent_module import Agent
import task_mgsm

def test_task_solving():
    agent = Agent()
    task = task_mgsm.MGSM_Task()
    
    # Test a single math problem
    test_problem = "If John has 5 apples and gives 2 to Mary, how many does he have left?"
    
    result = agent.solver(test_problem)
    print("\nTest Problem:", test_problem)
    print("Solution:", result)
    
    # Test task evaluation
    feedback, acc = task.evaluate(agent.solver)
    print("\nTask Evaluation:")
    print("Accuracy:", acc)
    print("\nFeedback Preview:")
    print(feedback[:500])

if __name__ == "__main__":
    test_task_solving() 