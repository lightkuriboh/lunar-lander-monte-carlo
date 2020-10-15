
from gymAI.constants.constants import *
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ENV_NAME = 'CartPole-v0'
ENV_NAME = 'LunarLander-v2'


def get_model_name():
    return 'trained_models/MonteCarlo'


def get_trained_model_name(idx):
    return '{0}-{1}'.format(get_model_name(), idx)


class MonteCarlo:
    """ Implementation of Monte-Carlo policy gradient """

    def __init__(self, num_actions, num_features):
        super(MonteCarlo, self).__init__()
        self.num_actions = num_actions

        self.tf_observations = tf.compat.v1.placeholder(tf.float32, [None, num_features],
                                                        name='observations')
        self.tf_values = tf.compat.v1.placeholder(tf.float32, [None, ], name='state_values')
        self.tf_actions = tf.compat.v1.placeholder(tf.int32, [None, ], name='num_actions')

        self.dense_layer1 = tf.compat.v1.layers.dense(
            inputs=self.tf_observations,
            units=HIDDEN_SIZE,
            activation=tf.nn.tanh
        )

        self.dense_layer2 = tf.compat.v1.layers.dense(
            inputs=self.dense_layer1,
            units=2 * HIDDEN_SIZE,
            activation=tf.nn.tanh
        )

        self.dense_layer3 = tf.compat.v1.layers.dense(
            inputs=self.dense_layer2,
            units=self.num_actions,
            activation=None
        )

        self.action_probabilities = tf.nn.softmax(self.dense_layer3)

        negative_log_probabilities = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.dense_layer3,
            labels=self.tf_actions
        )
        self.loss = tf.reduce_mean(negative_log_probabilities * self.tf_values)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.saver = tf.compat.v1.train.Saver()

    def improve_policy(self, actions, observations, cumulative_rewards):
        _, __ = self.sess.run(
            [self.optimizer, self.loss],
            feed_dict={
                self.tf_actions: np.array(actions),
                self.tf_values: np.array(cumulative_rewards),
                self.tf_observations: np.vstack(observations)
            }
        )
        pass

    def get_action_from_state(self, state):
        probabilities = self.sess.run(
            self.action_probabilities,
            feed_dict={
                self.tf_observations: state[np.newaxis, :]
            }
        )
        my_action = np.random.choice(range(probabilities.shape[1]), p=probabilities.ravel())
        return my_action

    def save_policy(self, idx):
        print("Saving checkpoint for version {} at: {}".format(idx, get_model_name()))
        self.saver.save(self.sess, get_model_name(), global_step=int(idx))
        pass

    def restore_policy(self, idx):
        self.saver.restore(self.sess, get_trained_model_name(int(idx)))
        pass


class Agent:
    def __init__(self, env, version=0):
        super(Agent, self).__init__()
        self.env = env
        self.num_actions = self.env.action_space.n

        self.rewards = list()
        self.actions = list()
        self.states = list()

        self.policy = MonteCarlo(num_actions=env.action_space.n, num_features=env.observation_space.shape[0])
        if version != 0:
            self.policy.restore_policy(version)

    def store_model(self, idx):
        self.policy.save_policy(idx)
        pass

    def get_action(self, state):
        return self.policy.get_action_from_state(state)

    def get_values(self):
        discounted_rewards = np.zeros_like(self.rewards)
        for t in range(len(self.rewards) - 1, -1, -1):
            discounted_rewards[t] = self.rewards[t]
            if t < len(self.rewards) - 1:
                discounted_rewards[t] += GAMMA * discounted_rewards[t + 1]

        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        return discounted_rewards

    def learn(self):
        self.policy.improve_policy(self.actions, self.states, self.get_values())
        pass

    def store_info(self, reward, action, state):
        self.rewards.append(reward)
        self.actions.append(action)
        self.states.append(state)

    def replay(self):
        self.rewards = []
        self.actions = []
        self.states = []


def calculate_episode_number(e, version):
    return int(e + version * SAVING_INTERVAL + 1)


def train(episode, version = 0):
    env = gym.make(ENV_NAME)
    env.seed(0)
    np.random.seed(0)

    scores = []
    steps = []
    agent = Agent(env, version)

    for e in range(episode):
        state = env.reset()
        score = 0

        for i in range(LIMIT_TURN):
            if IS_RENDER:
                env.render()

            action = agent.get_action(state)
            # action = 0

            next_state, reward, done, _ = env.step(action)

            agent.store_info(reward, action, state)

            state = next_state

            score += reward

            if done:
                agent.learn()
                agent.replay()

                steps.append(i)

                if IS_RENDER:
                    print("Score: {}, Steps: {}".format(score, i))
                break
        scores.append(score)

        real_episode = calculate_episode_number(e, version)

        if real_episode % MONITORING_INTERVAL == 0:
            print("Episode: {}/{}, average score over {} episodes: {}".format(
                real_episode,
                calculate_episode_number(version, 0) + episode,
                MONITORING_INTERVAL,
                np.mean(scores[-MONITORING_INTERVAL:]))
            )

        if not IS_RENDER and real_episode % SAVING_INTERVAL == 0:

            average_score = np.mean(scores[-SAVING_INTERVAL:])
            average_steps = np.mean(steps[-SAVING_INTERVAL:])

            agent.store_model(real_episode / SAVING_INTERVAL)

            print()
            print("Average score over last {0} episode(s): {1:.2f}".format(SAVING_INTERVAL, average_score))
            print()
            print("Average steps over last {0} episode(s): {1:.2f} \n".format(SAVING_INTERVAL, average_steps))
            print()

            if average_score > TERMINAL_SCORE:
                print('\n Task Completed! \n')
                break

    return scores


if __name__ == '__main__':

    loss_overall = train(EPISODE_NUMBER)
    # loss_overall = train(EPISODE_NUMBER, 11)
    plt.plot([i + 1 for i in range(0, len(loss_overall), 2)], loss_overall[::2])
    plt.show()


# Version 4 is very good
# Version 5 is also very good
# Version 6 is the best one
# Version 7 is good for score but toxic for fuel usage
