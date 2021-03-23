import gym
import numpy as np
import ray
from really import SampleManager
from gridworlds import GridWorld
import tensorflow as tf
import sys
import os

"""
Define global parameters for this project
"""
projectName = ""
savePath = ""


"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""


class TabularQ(object):
    def __init__(self, h, w, action_space):
        """
        Parameters:
            [int] h: env.height
            [int] w: env.width
            [int] action_space: count of all possible moves = 4
        """

        # action space consist of four possible moves
        # 0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"
        self.action_space = action_space
        self.h = h
        self.w = w

        ## # TODO:
        self.alpha = 0.9
        self.epsilon = 0.95
        self.gamma = 0.75

        # set the
        # self.Q = np.random.uniform(size=(1, action_space, h, w)) # 1 is the pseudo batch
        self.Q = np.zeros((1, action_space, h, w)) # 1 is the pseudo batch


    def __call__(self, state):
        ## # TODO:
        output = {}

        # select Q-value of the state pos
        output["q_values"] = self.Q[:, :, int(state[0,0]), int(state[0,1])]

        return output

    # # TODO:
    def get_weights(self):
        return self.Q

    def set_weights(self, weight):


        self.Q = weight

        #if state[0,0] == 0 and state[0,1] == 3:
        #    print(f"Q-Values is: {self.Q} with action: {action}")


    # what else do you need?
    def max_q(self, state):
        """
        Using the initalized values for Q we can calculate
        the Q value for a state and its action since the following
        state is dictated by the given action

        Params:
            (tuple) s: state
            (val) a: action indicated by an int

        Return:
            Q: where
        """

        Q = self.Q[0, np.argmax(self.Q[0, :, state[0, 0], state[0, 1]]), state[0, 0], state[0, 1]]

        return Q

    def greedy_action(self, state):
        """
        Gets the Q values from the agent an tries to find the best
        fitting policy using greedy method.
        To calculate the

        Params:
            state:  since we want to select the best fitting action
                    we need a starting point
        Returns:
            (val) A: containing the best fitting action calculated by
                     selecting the most rewarding Q-Value
        """

        A = np.argmax(self.Q[0, :, state[0, 0], state[0, 1]])

        return A

    def save_model(self, saving_path, episode):
        # create a log file containing values:
        # - Q

        pass


if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 10,
        "width": 10,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
    }

    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 2,
        "total_steps": 100,
        "model_kwargs": model_kwargs
        # and more
    }


    # initilize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    print("test before training: ")
    manager.test(
        max_steps=100,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )


    saving_path = os.getcwd() + "/progress_test"


    episodes = 20
    saving_after = 5
    max_steps = 100

    agent = manager.get_agent()


    for e in range(episodes):
        state = np.expand_dims(env.reset(), axis=0)
        print(f"episode: {e}")

        for t in range(max_steps):
            env.render()

            # choose a from S using policy derivied from Q (greedy)
            action = int(agent.act(state))

            # Take action A and observe R,S'
            state_new, reward, done, _ = env.step(action)
            state_new = np.expand_dims(state_new, axis=0)

            weights = agent.model.Q.copy()

            # Q(S,A) ← Q(S,A) + a[R+γ maxQ(S',a)-Q(S,A)]
            weights[0, action, state[0, 0], state[0, 1]] = (
                weights[0, action, state[0, 0], state[0, 1]] +
                agent.model.alpha * (
                    reward + agent.model.gamma * agent.max_q(state_new) - weights[0, action, state[0, 0], state[0, 1]])
            )

            # update Q-values
            agent.model.set_weights(weights)

            # setting the state
            state = state_new

            #if t % 50 == 0:
                # print(agent.model.Q)

            if done:
                print("Reached Goal")
                print(agent.model.Q)
                break


    env.close()
