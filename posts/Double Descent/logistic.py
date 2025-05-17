import torch
import math

class LinearModel:
    def __init__(self):
        self.w = None
    
    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None:
            self.w = torch.rand((X.size()[1]))

        # Matrix multiply
        return X @ self.w
    
    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        scores = self.score(X)
        return (scores > 0).float()

class Perceptron(LinearModel):
    
    def loss(self, X, y):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        HINT: You are going to need to construct a modified set of targets and predictions that have entries in {-1, 1} -- otherwise none of the formulas will work right! An easy to to make this conversion is: 
        
        y_ = 2*y - 1
        """
        scores = self.score(X)
        y_ = 2 * y - 1
        misclassified = 1.0 * (y_ * scores < 0)
        return misclassified.mean()
    
    def grad(self, X, y):
        """
         Compute gradient of the misclassification loss with respect to model weight vector (w).
 
         Identifies misclassified examples using modified labels (mapped from {0, 1} to {-1, 1})
         and then computes the update vector based on those misclassifications.
 
         ARGUMENTS:
             X, torch.Tensor: the feature matrix. X.size() = (n, p), where n is number of data points,
             and p is number of features. Assumes the last column is a bias term of 1s.
 
             y, torch.Tensor: the target vector. y.size() = (n,). Entries should be 0 or 1.
 
         RETURNS:
             torch.Tensor: Vector of size (p,) representing the gradient update to be applied to w.
        """
        score = X @ self.w
        y_ = 2 * y - 1
        misclassified = 1.0 * (y_ * score < 0).float()
        # Had to reshape both to [300,1] tensors to multiply
        # https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html
        misclassified = misclassified.reshape(misclassified.shape[0], 1)
        y_ = y_.reshape(y_.shape[0], 1)


        # Return update
        return torch.mean((- (misclassified * y_) * X), dim = 0)

class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        loss = self.model.loss(X, y)
        """
        Params of randint: https://pytorch.org/docs/stable/generated/torch.randint.html
        low (lowest integer to bound selection)
        high (One above the highest integer to be drawn from the distribution)
        size (tuple defining shape of output tensor)

        X.size() returns tuple of dimensions of tensor. We want to select row (i.e. data point/entry)
        .item() converts the tensor to python integer
        """ 
        
        grad_mini = self.model.grad(X, y)
        self.model.w = self.model.w - grad_mini
        return loss
    
class LogisticRegression(LinearModel):

    def loss (self, X, y):
        """
        Compute the average logistic loss across the dataset.

        Applies sigmoid function to model scores and computes a value 
        for each prediction.

        ARGUMENTS:
            X, torch.Tensor: Feature matrix. Size == (n, p).
            y, torch.Tensor: Binary target labels. Size == (n,).

        RETURNS:
            torch.Tensor: A scalar tensor representing the mean logistic loss.
        """
        scores = self.score(X)

        # Turn scores into 0-1 values with sigmoid function
        # https://pytorch.org/docs/stable/generated/torch.exp.html
        # Returns e ** x, where you pass x as exponent
        sigmoid_scores = 1 / (1 + torch.exp(-scores))


        # Need to avoid log(0) bc it is nan
        # .clamp() restricts vals within range - don't go above min or below max
        # 1e-7 is the minimum offest my jupyter notebook works with (i.e. doesn't return nan)
        offset_min = 1e-7
        sigmoid_scores = sigmoid_scores.clamp(min = offset_min, max = 1 - offset_min)

        # Compute logistic loss (binary cross entropy)
        L = -y * torch.log(sigmoid_scores) - ((1 - y) * torch.log(1 - sigmoid_scores))

        # Return average loss over all data points
        return L.mean()
    
    def grad(self, X, y):
        """
        Compute the gradient of the logistic loss.

        Gradient is computed using the difference between predictions
        and true labels, scaled by the input features.

        ARGUMENTS:
            X, torch.Tensor: Feature matrix. Shape (n, p).
            y, torch.Tensor: Binary target labels. Shape (n,).

        RETURNS:
            torch.Tensor: A gradient vector of shape (p,) for weight updates.
        """
        scores = self.score(X)
        # Convert scores with sigmoid function
        sigmoid_scores = 1 / (1 + torch.exp(-scores))

        # Calculate prediction error (how far are we from true label)
        error = sigmoid_scores - y

        # Reshape error to facilitate pytorch multiplication
        error_ = error[:, None]

        # Multiply each error by its corresponding input features
        # This gives gradient direction
        gradient = error_ * X

        # Return average gradient across all examples
        return torch.mean(gradient, dim = 0)

class GradientDescentOptimizer:
    
    def __init__(self, model):
        self.model = model
        # Store previous weight vector
        self.prev_w = None

    def step(self, X, y, alpha, beta):
        """
        Performs a single step of gradient descent (with momentum)

        Input:
            X: feature matrix (n, p)
            y: target value vector (n,)
            alpha: learning rate
            beta: momentum
        """

        # Initialize weights if haven't been set
        if self.model.w is None:
        # Trigger the model to initialize weights using score() if doesn't already exist
            self.model.score(X)

        # If first step, store copy of OG weight vector
        if self.prev_w is None:
            self.prev_w = self.model.w.clone()
        
        # Compute the gradient of the loss
        grad = self.model.grad(X, y)

        # Calculate new weight vector using gradient descent with momentum
        # Momentum term "Remembers" previous step
        new_w = self.model.w - alpha * grad + beta * (self.model.w - self.prev_w)
        
        # Store curr weights as previous for next iteration
        self.prev_w = self.model.w.clone()

        # Update model weights
        self.model.w = new_w

class MyLinearRegression(LinearModel):

    def predict(self, X):
        """
        Predicts scores for input feature matrix X

        Input:
            X: feature matrix. Shape (num_samples, num_features)
        
        Returns:
            Tensor of predicted outputs
        """
        return self.score(X)
    
    def loss(self, X, y):
        """
        Computes mean squared error loss between predictions and targets

        Input:
            X: Feature matrix
            y: target vector

        Returns:
            Average MSE loss
        """
        preds = self.predict(X)
        return torch.mean((preds - y) ** 2)


class OverParameterizedLinearRegressionOptimizer:

    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        """
        Fits model weights to data using Moore-Penrose pseudoinverse

        Input:
            X: Feature matrix
            y: target vector
        """
        self.model.w = torch.linalg.pinv(X) @ y