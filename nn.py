class IntentNN:
    def __init__(self, input_size, hidden_size, output_size):
        import random
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        self.W2 = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]

    def _sigmoid(self, x):
        return 1 / (1 + pow(2.71828, -x))

    def _softmax(self, x):
        exps = [pow(2.71828, v) for v in x]
        s = sum(exps)
        return [v/s for v in exps]

    def forward(self, x):
        h = [0]*self.hidden_size
        for j in range(self.hidden_size):
            h[j] = sum(x[i]*self.W1[i][j] for i in range(self.input_size))
            h[j] = self._sigmoid(h[j])

        y = [0]*self.output_size
        for k in range(self.output_size):
            y[k] = sum(h[j]*self.W2[j][k] for j in range(self.hidden_size))
        return h, self._softmax(y)

    def train(self, x, target, lr=0.1):
        h, y = self.forward(x)

        error = [target[k] - y[k] for k in range(self.output_size)]

        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.W2[j][k] += lr * error[k] * h[j]

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                grad = h[j]*(1-h[j]) * sum(error[k]*self.W2[j][k] for k in range(self.output_size))
                self.W1[i][j] += lr * grad * x[i]

    def predict(self, x, labels):
        _, y = self.forward(x)
        idx = y.index(max(y))
        return labels[idx], y[idx]
