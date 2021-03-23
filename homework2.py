
import logging, os
from tensorflow.python.ops.gradients_impl import gradients

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import gym
import ray
from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets

"""
Questions:
- to create a second agent would I have to create a new manager?
"""



class DQN(tf.keras.Model):
    """
    Approximates the Q-value of the current state transition given a
    state tuple

    Params:
        (tuple) state:
            gmy environment encoded as a tuple

    Returns:
        (dict) output:
            key: q_values
            value: approximation of the q_value
    """

    def __init__(self):

        super(DQN, self).__init__()

        self.layer1 = tf.keras.layers.Dense(128, activation="relu")
        self.layer2 = tf.keras.layers.Dense(64, activation="relu")
        self.layerOut = tf.keras.layers.Dense(1, activation=None)

    def call(self, state):

        output = {}

        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layerOut(x)

        output["q_values"] = x
        return output


class Duelling(tf.keras.Model):
    """
    Approximate the advantage for each action and additionally
    estimate a value score for the current action

    Params:
        (tuple) state:
            gmy environment encoded as a tuple

    Returns:
        (dict) output:
            key: q_values
            value: approximation of the q_value
    """

    def __init__(self):

        super(Duelling, self).__init__()

        self.layer1 = tf.keras.layers.Dense(32, activation="relu")
        self.layer2 = tf.keras.layers.Dense(64, activation="relu")

        self.value = tf.keras.layers.Dense(1, activation=None)
        self.advantage = tf.keras.layers.Dense(2, activation=None)

    def call(self, state):
        output = {}

        x = self.layer1(state)
        x = self.layer2(x)

        value = self.value(x)
        advantage = self.advantage(x)

        Q_Value = value + advantage - tf.math.reduce_mean(advantage, keepdims=True)

        output["q_values"] = Q_Value
        return output


def new_epsilon(time, eps, A=0.5, B=0.1, C=0.1):
    standardized_time=(time- A*eps)/(B*eps)
    cosh=np.cosh(np.exp(-standardized_time))
    epsilon=1.1-(1/cosh+(time*C/eps))
    return epsilon

if __name__ == "__main__":

    kwargs = {
        "model": Duelling,
        "environment": "CartPole-v0",
        "num_parallel": 5,
        "total_steps": 100,
        "action_sampling_type": "epsilon_greedy",
        "num_episodes": 20,
        "epsilon": 1,
    }

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_test"

    buffer_size = 50000
    test_steps = 160
    epochs = 20
    sample_size = 10000
    optim_batch_size = 8
    saving_after = 5
    gamma = 0.90

    optimizer = tf.keras.optimizers.Adam()

    errorMessage = "Pleas change the test_steps to a muliple of batch size"
    assert test_steps % optim_batch_size == 0, errorMessage

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize experience replay buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    # manager.test(test_steps, do_print=True, render=True)

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):
        print(f"starting epoch: {e}")

        # experience replay
        # print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)


        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        # print(f"collected data for: {sample_dict.keys()}")

        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        # print("optimizing...")
        packedValues = list(data_dict.values())
        packedValues = zip(*packedValues)

        average_loss = []

        for state, action, reward, stateNew, done in packedValues:

            tempLoss = []
            with tf.GradientTape() as tape:
                prediction = tf.cast(agent.max_q(stateNew), dtype=tf.float32)

                # compute the target value for Q
                Q_Target = reward + gamma * prediction * done

                # calculate the loss from the target and estimated Q-Value
                loss = (Q_Target - agent.q_val(state, action))**2

                tempLoss.append(loss)

            gradients = tape.gradient(loss, agent.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))


            average_loss.append(tempLoss)


        # get and set new weights globally
        new_weights = agent.model.get_weights()
        manager.set_agent(new_weights)

        # get new weights
        agent = manager.get_agent()

        # update aggregator
        time_steps = manager.test(test_steps)

        try:
            manager.update_aggregator(loss=average_loss, time_steps=time_steps)
        except Exception:
            print(average_loss)
            print(time_steps)
            raise

        # print progress
        print(f"epoch ::: {e}", end="   ")
        print(f"loss ::: {np.mean([np.mean(l) for l in average_loss])}", end="   ")
        print(f"avg env steps ::: {np.mean(time_steps)}")

        # yeu can also alter your managers parameters
        manager.set_epsilon(new_epsilon(e, epochs))

        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)

    # and load mmodels
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
