import numpy as np


class EnsembleAgent:
    def __init__(self, agents):
        self.agents = agents

    def predict(self, obs, deterministic=True):
        predictions = np.array(
            [agent.predict(obs, deterministic)[0] for agent in self.agents]
        )
        values, counts = np.unique(predictions, return_counts=True)
        return (values[np.argmax(counts)],)
