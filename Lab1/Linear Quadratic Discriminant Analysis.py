#%%
import numpy as np
import matplotlib.pyplot as plt
import util

#%%
def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    
    # Sort data into male and female
    f_data = np.empty((2,))
    m_data = np.empty((2,))
    
    for i in range( y.shape[0] ):
        # Male
        if y[i] == 1:
            m_data = np.vstack( ( m_data, x[i]) )
        # Female
        if y[i] == 2:
            f_data = np.vstack( (f_data, x[i]) )
    
    # Trim first rows of f_data and m_data since theyre not valid
    m_data = np.delete(m_data, (0), axis=0)
    f_data = np.delete(f_data, (0), axis=0)
        
    # Get mean for height and weight by sex
    mu_female = np.sum(f_data, axis=0) / f_data.shape[0] # data.shape[0] is the number of data points
    mu_male = np.sum(m_data, axis=0) / m_data.shape[0]
    mu = np.sum(x, axis=0) / x.shape[0]
    
    # Get covariance 
    cov_male = np.matmul( (m_data - mu_male).T, (m_data - mu_male) ) / m_data.shape[0]
    cov_female = np.matmul( (f_data - mu_female).T, (f_data - mu_female) ) / f_data.shape[0]
    cov = np.matmul( (x - mu).T, (x - mu) ) / x.shape[0]
    
    # Classifications    
    PlotLDA( m_data, f_data, mu_male, mu_female, cov, 0.5)
    PlotQDA(m_data, f_data, mu_male, mu_female, cov_male, cov_female, 0.5)
    
    return (mu_male,mu_female,cov,cov_male,cov_female)

#%%
def PlotLDA( m_data, f_data, mu_male, mu_female, cov, pi ):
    
    # Plot male and female features for reference
    plt.figure(0)
    plt.scatter(m_data[:, 0], m_data[:, 1], c="blue", s=7, label="male")
    plt.scatter(f_data[:, 0], f_data[:, 1], c="red", s=7, label="female")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.legend(loc="upper right")
    plt.title("Linear Discriminant Analysis: Height vs Weight")

    # Get Gaussian density to plot contour
    # Create an array with every combination of height and weight
    height = np.arange(55,80)
    weight = np.arange(75,300)
    hw = []
    for w in weight:
        for h in height:
            hw.append( [h,w] )
    
    m_gaussian = util.density_Gaussian(mu_male, cov, np.asarray(hw)).reshape(len(weight), len(height))
    f_gaussian = util.density_Gaussian(mu_female, cov, np.asarray(hw)).reshape(len(weight), len(height))
    
    # Plot the contours
    colors=['#383838', '#BEBEBE', '#696969']
    plt.figure(0)
    plt.contour( height, weight, m_gaussian, levels=3, colors=colors )
    plt.contour( height, weight, f_gaussian, levels=3, colors=colors )
    
    # Get boundaries and decisions
    m = LDA(mu_male, cov, hw, pi)
    f = LDA(mu_female, cov, hw, 1-pi)
    decision = (m - f).reshape( len(weight), len(height) )
    
    plt.figure(0)
    plt.contour( height, weight, decision, levels=0)
    
#%%
def LDA(mu, cov, x, pi):
    return np.log(pi) - 0.5 * (mu.T @ np.linalg.inv(cov) @ mu) + (x @ np.linalg.inv(cov) @ mu.T)
    
    
#%%
def PlotQDA(m_data, f_data, mu_male, mu_female, cov_male, cov_female, pi):
    
    # Plot male and female features for reference
    plt.figure(1)
    plt.scatter(m_data[:, 0], m_data[:, 1], c="blue", s=7, label="male")
    plt.scatter(f_data[:, 0], f_data[:, 1], c="red", s=7, label="female")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.legend(loc="upper right")
    plt.title("Quadratic Discriminant Analysis: Height vs Weight")
    
    # Get Gaussian density to plot contour
    # Create an array with every combination of height and weight
    height = np.arange(55,80)
    weight = np.arange(75,300)
    hw = []
    for w in weight:
        for h in height:
            hw.append( [h,w] )
    
    hw =  np.asarray(hw)
    
    m_gaussian = util.density_Gaussian(mu_male, cov_male, hw).reshape(len(weight), len(height))
    f_gaussian = util.density_Gaussian(mu_female, cov_female, hw).reshape(len(weight), len(height))
    
    # Plot the contours
    colors=['#383838', '#BEBEBE', '#696969']
    plt.figure(1)
    plt.contour( height, weight, m_gaussian, levels=3, colors=colors )
    plt.contour( height, weight, f_gaussian, levels=3, colors=colors )
    
    # Get boundaries and decisions
    decision = []
    for i in range( hw.shape[0] ):
        m = QDA(mu_male, cov_male, hw[i], pi)
        f = QDA(mu_female, cov_female, hw[i], 1-pi)
        decision.append(m-f)
    
    decision = np.asarray(decision).reshape( len(weight), len(height) )
    print(decision)
    print(decision.shape)
        
    
    plt.figure(1)
    plt.contour( height, weight, decision, levels=0 )

#%%
def QDA(mu, cov, x, pi):
    diff = x - mu
    return np.log(pi) - 0.5*np.log( np.linalg.det(cov) ) - 0.5*np.sum( diff @ np.linalg.inv(cov) @ diff.T ) 
    
#%%
def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    pi = 0.5
    
    # LDA
    LDAmiss = 0
    for i in range( len(y) ):
        # Predict male
        if LDA( mu_male, cov, x[i,:], pi ) > LDA( mu_female, cov, x[i,:], pi ):
            if y[i] == 2:
                LDAmiss += 1
        
        # Predict female
        else:
            if y[i] == 1:
                LDAmiss += 1
    
    # Get misclassification in percent
    LDAmiss = ( LDAmiss / len(y) ) * 100
    
    
    # QDA
    QDAmiss = 0
    for i in range( len(y) ):
        #Predict male
        if QDA(mu_male, cov_male, x[i,:], pi) > QDA(mu_female, cov_female, x[i,:], pi):
            if y[i] == 2:
                QDAmiss += 1
        else:
            if y[i] == 1:
                QDAmiss += 1
    
    # Get misclassification in percent
    QDAmiss = ( QDAmiss / len(y) ) * 100
    
    return LDAmiss, QDAmiss

#%%
if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA, mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    
    print( "LDA:", mis_LDA )
    print( "QDA:", mis_QDA )

