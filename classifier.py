# Amir Lorvand
# Classifier: MLP

import numpy as np 

class Classifier:
    def __init__(self):
        # hyperparameters
        self.input_dim = 25
        self.hidden_dim = 32
        self.output_dim = 4
        self.learning_rate = 0.1
        self.epochs = 200

        # model parameters
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        
        self.trained = False
        np.random.seed(35)

    def reset(self):
        # reset to initial phase
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.trained = False
    
    def fit(self, data, target):
        # prepare data - shape X = (N, 25) y = (N,)
        X = np.array(data, dtype=np.float32)
        y = np.array(target, dtype=np.int32)
        N = X.shape[0]

        # one-hot encoding targets
        Y = np.zeros((N, self.output_dim), dtype=np.float32)
        Y[np.arange(N), y] = 1.0

        # initialise weights
        self.W1 = 0.01 * np.random.randn(self.input_dim, self.hidden_dim)
        self.b1 = np.zeros((1, self.hidden_dim), dtype=np.float32)
        self.W2 = 0.01 * np.random.randn(self.hidden_dim, self.output_dim)
        self.b2 = np.zeros((1, self.output_dim), dtype=np.float32)

        # training using gradient descent
        for epoch in range(self.epochs):

            #forward pass
            probs, cache = self._forward(X)
            Xc, Z1, A1, Z2 = cache

            # cross entropy loss function
            eps = 1e-12
            loss = -np.sum(Y * np.log(probs+eps)) / N

            # print loss every 50 epoc
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

            # backpropagation 
            derivative_Z2 = (probs - Y) / N
            derivative_W2 = A1.T.dot(derivative_Z2)
            derivative_b2 = np.sum(derivative_Z2, axis=0, keepdims=True)

            # backpropagation - hidden layer 
            derivative_A1 = derivative_Z2.dot(self.W2.T)
            derivative_Z1 = derivative_A1 * self._relu_derivative(Z1)

            # gradients for W1 & b1
            derivative_W1 = Xc.T.dot(derivative_Z1)
            derivative_b1 = np.sum(derivative_Z1, axis=0, keepdims=True)

            # gradient descent update
            self.W1 -= self.learning_rate * derivative_W1
            self.b1 -= self.learning_rate * derivative_b1
            self.W2 -= self.learning_rate * derivative_W2
            self.b2 -= self.learning_rate * derivative_b2
                
        self.trained = True

    # ReLu activation function, return max(0, z)
    def _relu(self, z):
        return np.maximum(0.0, z)
    
    # derivative of ReLu
    def _relu_derivative(self, z):
        return (z > 0).astype(np.float32)
    
    # softmax function, convertis output into probabilities that sum to 1
    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # forward pass through the network
    def _forward(self, X):
        Z1 = X.dot(self.W1) + self.b1 
        A1 = self._relu(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        output = self._softmax(Z2)

        cache = (X, Z1, A1, Z2)
        return output, cache
    
    # predict an action from the feature vector
    def predict(self, data, legal=None):
        x = np.array(data, dtype=np.float32).reshape(1, -1) # convert x into shape of (1, 25)

        # return the default move is the model is nt trained
        if not self.trained:
            return 1
        
        probs, _ = self._forward(x) 
        return int(np.argmax(probs, axis=1)[0])

        
