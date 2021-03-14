import numpy as np

class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon
        self.states = []
        self.actions = []
        self.rewards = []

    def fit(self, env, steps=1000):
        """
        Trains the MultiArmedBandit on an OpenAI Gym environment.

        See page 32 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2018.pdf).
        Initialize your parameters as all zeros. For the step size (alpha), use
        1 / N, where N is the number of times the current action has been
        performed. Use an epsilon-greedy policy for action selection.

        See (https://gym.openai.com/) for examples of how to use the OpenAI
        Gym Environment interface.

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "done" returned
            from env.step() is True.
          - If all values of a np.array are equal, np.argmax deterministically
            returns 0.
          - In order to avoid non-deterministic tests, use only np.random for
            random number generation.
          - MultiArmedBandit treats all environment states the same. However,
            in order to have the same API as agents that model state, you must
            explicitly return the state-action-values Q(s, a). To do so, just
            copy the action values learned by MultiArmedBandit S times, where
            S is the number of states.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length 100.
            Let s = np.floor(steps / 100), then rewards[0] should contain the
            average reward over the first s steps, rewards[1] should contain
            the average reward over the next s steps, etc.
        """
        s = np.floor(steps/100)
        states = env.observation_space.n
        actions = env.action_space.n
        state_action_values = np.zeros((states, actions))

        env.reset()

        q = np.zeros(actions)
        n = np.zeros(actions)
        choices = []

        def greedy():
          decision = np.random.random() < self.epsilon
          if decision:
            return env.action_space.sample()
          else:
            return np.argmax(q)

        cum_rewards = []
        rewardz = []

        for episode_number in range(steps):
          decision = greedy() # decision is of type int
          observation, reward, done, info = env.step(decision)

          cum_rewards.append(reward)
          if episode_number % s == 0:
            rewardz.append(sum(cum_rewards)/len(cum_rewards))
            cum_rewards = []

          n[decision] += 1
          q[decision] = (1 - 1/n[decision]) * q[decision] + (1/n[decision]) * reward
          for row_count, _ in enumerate(state_action_values):
            state_action_values[row_count] = q
          choices.append(decision)

          if done:
            env.reset()

        return state_action_values, np.array(rewardz)

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the MultiArmedBandit algorithm and the state action values. Predictions
        are run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. Any mechanisms used for
            exploration in the training phase should not be used in prediction.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        """
        q = state_action_values[0]
        best = np.argmax(q)
        observation, reward, done, info = env.step(best)
        return np.array([0]), np.array([best]), np.array([reward])
        """
        sta, act, rew = [], [], []

        brek = False
        init = env.reset()
        
        while not brek:
          best = np.argmax(state_action_values[init])
          observation, reward, done, info = env.step(best)

          init = observation

          sta.append(observation)
          act.append(best)
          rew.append(reward)

          brek = done
        
        return np.array(sta), np.array(act), np.array(rew)