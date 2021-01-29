import numpy as np
import matplotlib.pyplot as plt
import util
#%%

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    
    mean = np.array( [0, 0] )
    cov = np.array( [ [beta, 0], [0, beta] ] )
    
    # Plot contours
    x, y, density = drawGaussianDensity(-1, 1, 100, mean, cov)

    plt.figure()
    plt.contour( x, y, density)
    plt.plot(-0.1, -0.5, 'ro', label='true a') # Given
    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.title("Prior Distribution of a")
    plt.legend()
    
    return 

#%%
    
def posteriorDistribution(x,z,beta,sigma2,n):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    n: number of training data points
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    # adding a col of 1 to the rightmost side of vector x to match cov dims
    x = np.column_stack( (x, np.ones((x.shape[0], 1)) ) )
    
    # Get covariance
    cov = np.linalg.inv( x.T @ x + (sigma2/beta)*np.identity(2) ) * sigma2
    
    # Get mean
    mean = ( ( cov @ (x.T @ z) ) / sigma2 )

    # Plot gaussian density contours
    x, y, density = drawGaussianDensity(-1, 1, 100, mean.T, cov )
    
    plt.figure()
    plt.contour( x, y, density)
    plt.plot(-0.1, -0.5, 'ro', label='true a') # Given
    plt.xlabel("a0")
    plt.ylabel("a1")
    title = "Posterior Distribution of a with " + str(n) + " data samples" 
    plt.title(title)
    plt.legend()

    return mean, cov

#%%

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train, n):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    # adding a col of 1 to the rightmost side of vector x to match cov dims
    x = np.asarray(x) #Upper case: augmented | Lower case: original
    X = np.column_stack( (x, np.ones((x.shape[0], 1)) ) )
    
    # Get prediction    
    z_mu = mu.T @ X.T
    
    z_covar = X @ Cov @ X.T + sigma2
    
    z_stdev = np.diag(z_covar) ** (1/2) # Variances are the diagonal elements
    
    #Plot
    plt.figure()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.scatter(x_train, z_train, label='Training Samples', c='#273469', s=9)
    plt.errorbar(x, z_mu.T, yerr=z_stdev, ecolor='#DB5461', elinewidth=2, linewidth=3, capsize=2, label='Predicted target and error')
    title = "Regression with " + str(n) + " data samples" 
    plt.title(title)
    plt.legend()

    
    
    return 
#%%
def drawGaussianDensity(lowerBound, upperBound, increments, mean, covariance):
    """
    Give back vectors that can be used to plot contours
    
    Inputs:
    ------
    lowerBound: lower bound of x and y axis
    upperBound: upper bound of x and y axis
    increments: number of increments between lowerBound and upperBound
    mean: (2,1) matrix
    cov: (2,2) matrix
    
    Outputs: X, Y, density that is passed to plt.contour
    -----
    """
    
    # Create inputs of meshgrid
    X =  np.linspace(lowerBound, upperBound, num=increments)
    Y=  np.linspace(lowerBound, upperBound, num=increments)
    
    # Create list of every x for every y to input into density_Gaussian
    xy = []
    for x in np.nditer(X):
        for y in np.nditer(Y):
            xy.append( [x, y] )
    
    # Get gaussian density 
    
    density = util.density_Gaussian( mean, covariance, np.asarray(xy) ).reshape( X.shape[0], Y.shape[0] )
    
    return X, Y, density
#%%

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # number of training samples used to compute posterior
    ns  = [1, 5, 100]
    
    for n in ns:
        # used samples
        x = x_train[0:n]
        z = z_train[0:n]

        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2,n)
        
        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z, n)
    

