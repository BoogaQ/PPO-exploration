import numpy as np

class FeedForwardNetwork(object):
    def __init__(self, env, hidden_sizes):
        self.env = env
        self.action_space = self.env.action_space.__class__.__name__
        self.weights = []

        layer_sizes = [env.observation_space.shape[0], *hidden_sizes, self.num_actions]
        for index in range(len(layer_sizes)-1):
            self.weights.append(np.zeros(shape=(layer_sizes[index], layer_sizes[index+1])))

    @property
    def num_actions(self):
        if self.action_space == "Discrete":
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        for i in range(len(self.weights)-1):
            out = np.dot(out, self.weights[i])
            out = np.arctan(out)

        out = np.dot(out, self.weights[-1])
        out = self.get_action(out.astype(float))
        return out

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)
    
    def softmax(self, logits):
        return np.exp(logits)/np.sum(np.exp(logits))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def discrete_action(self, n_actions, logits):
        return np.random.choice(np.arange(n_actions), p=self.softmax(logits).squeeze()).astype(int)

    def continuous_action(self, logits):
        mean = logits
        sd = np.ones(shape=mean.shape)
        return np.random.normal(mean, sd)
        #return mean.squeeze()

    def get_action(self, logits):
        if self.action_space == "Discrete":
            return self.discrete_action(self.env.action_space.n, logits)
        else:
            return self.continuous_action(logits)
