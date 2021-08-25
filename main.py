import numpy as np

# Linear Regression Class
class LinearRegression:
    
    def fit(self, X, y, epoch = 1500, learning_rate = 0.1, r = 0.1):
        """
        Caculates the parameters (theta) for the models using Gradient Descent.
        Params : 
            X : numpy array (NumberOfFeatures * NumberOfInstances) : Features
            y : row vector (NumberOfInstances * 1) : Lables
            epoch : int : number of iterations
            learning_rate : int
            r : regularization factor (not used yet)
        Returns :
            theata : A row vector (1 * (NumberOfFeatures + 1)) consists of parameters for each feature
        """
        m, n = X.shape
        self.theta = np.zeros((n + 1, 1))

        X = np.c_[np.ones((m, 1)), X] 
        

        for i in range(epoch):
            grad = 1 / m * X.T.dot(X.dot(self.theta) - y)
            cost = 1 / (2 * m) * np.sum(np.square(X.dot(self.theta) - y))
            self.theta = self.theta - grad * learning_rate

            print(f"Epoch {i + 1}, cost {cost}")

        return self.theta

    def predict(self, X):
        m, n = X.shape
        X = np.c_[np.ones((m, 1)), X]
        return X.dot(self.theta)

temp = LinearRegression()
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

temp.fit(X, y)

X_new = np.array([[0], [2]])

print(temp.predict(X_new))