import numpy as np
import gym
import random
import matplotlib.pyplot as plt

def train_and_get_num_steps(learning_rate,discount_rate):
    # Create Taxi environment
    env = gym.make('Taxi-v3')

    # Initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # Hyperparameters
    epsilon = 1.0
    decay_rate = 0.005
    max_steps = 99  # per episode
    max_change_threshold = 0.001

    numSteps = 0
    converged = False
    old_qtable = np.copy(qtable)

    # Training
    while True:
        state = env.reset()
        done = False
        numSteps += 1

        for s in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, done, info = env.step(action)

            new_q_state = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])
            qtable[state, action] = new_q_state
            state = new_state

            if done:
                break

        max_diff = np.max(np.abs(qtable - old_qtable))
        old_qtable = np.copy(qtable)

        if max_diff < max_change_threshold:
            converged = True
            break

        epsilon = epsilon * (1 - decay_rate)

    return numSteps

if __name__ == "__main__":
    learning_rates = np.linspace(0.1, 0.99, 40)
    numSteps_list = []

    for lr in learning_rates:
        numSteps = train_and_get_num_steps(lr,0.8)
        numSteps_list.append(numSteps)

    # Create a table with learning rates and corresponding numSteps
    table = zip(learning_rates, numSteps_list)

    # Print the table
    headers = ["Learning Rate", "Number of Steps (numSteps)"]
    print(tabulate(table, headers, tablefmt="pipe"))

    # Plot the graph
    plt.plot(learning_rates, numSteps_list)
    plt.xlabel('Learning Rate')
    plt.ylabel('Number of Steps (numSteps)')
    plt.title('Learning Rate vs. Number of Steps')
    plt.show()


    discount_rates = np.linspace(0.1, 0.99, 40)
    numSteps_list = []

    for dr in discount_rates:
        numSteps = train_and_get_num_steps(0.9,dr)
        numSteps_list.append(numSteps)

    # Create a table with learning rates and corresponding numSteps
    table = zip(discount_rates, numSteps_list)

    # Print the table
    headers = ["Discount Rate", "Number of Steps (numSteps)"]
    print(tabulate(table, headers, tablefmt="pipe"))

    # Plot the graph
    plt.plot(discount_rates, numSteps_list)
    plt.xlabel('Discount Rate')
    plt.ylabel('Number of Steps (numSteps)')
    plt.title('Discount Rate vs. Number of Steps')
    plt.show()
