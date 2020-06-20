import numpy as np

class Perceptron(object):
    
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self,X,Y):
        
        Initializer = np.random.RandomState(self.random_state)
        self.w_ = Initializer.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,Y):
                update = self.eta * (target - self.prediction(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def prediction(self,X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    #def getError(self):
    #    return self.errors_
    
class AdalineGD(object):
    def __init__(self,eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self,X,Y):
        Initializer = np.random.RandomState(self.random_state)
        self.w_ = Initializer.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (Y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def activation(self,X):
        return X
    
    def prediction(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
class AdalineSGD(object):
    def __init__(self,eta = 0.01, n_iter = 50,shuffle = True ,random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self,X,Y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            if self.shuffle:
                X,Y = self._shuffle(X,Y)
            cost = []
            for xi, target in zip(X,Y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost) / len(Y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self,X,Y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,Y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,Y)
    
    def _shuffle(self,X,Y):
        r = self.Initializer.permutation(len(Y))
        return X[r], Y[r]
    
    def _initialize_weights(self,m):
        self.Initializer = Initializer = np.random.RandomState(self.random_state)
        self.w_ = Initializer.normal(loc = 0.0, scale = 0.01, size = 1 + m)
        self.w_initialized = True
        
    def _update_weights(self,xi,target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = (error ** 2) * 0.5
        return cost
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def activation(self,X):
        return X
    
    def prediction(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)