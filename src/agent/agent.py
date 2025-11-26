class Agent:
    def __init__(self, name):
        self.name = name

    def act(self, observation):
        raise NotImplementedError("Subclasses must implement this method")
