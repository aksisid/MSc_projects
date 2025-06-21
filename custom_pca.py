#Add 90 for random_state
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000,n_features=20,centers=3,cluster_std=1.0,random_state=90)

#PCA Function
def custom_pca(data, keep_dims=1, method='eig'):
    
    import numpy as np
    
    if method=='eig':
        print('Eigen decomposition is used...')
        
         #Center the data
        X_centered = (data - np.mean(data, axis=0)) / np.std(data, axis=0) #axis=0 choosing all features     
                      
        #Calculate the covariance matrix
        num_samples = X_centered.shape[0] #gives number of rows/samples
        S_covmatrix = np.dot(X_centered.T, X_centered) / (num_samples - 1)
        
        #Compute eigenvalues and eigenvectors
        eig_vals, eig_vecs = np.linalg.eig(S_covmatrix)
        Lamda = eig_vals
        Q = eig_vecs
        
        #sort indices of eig_val based on their values(keeps record)
        sort_indices = np.argsort(Lamda)[::-1]
        
        #Sort eigenvalues and eigenvectors in descending order
        sort_Lamda = Lamda[sort_indices]
        sort_Q = Q[:, sort_indices]
        
        #Keep only the top eigenvectors (select principal components)
        Q_top = sort_Q[:, :keep_dims] #top keep_dims components
        
        #Project the data
        X_projected = np.dot(X_centered, Q_top)
    
        #Compute explained variance
        explained_variance = sort_Lamda / np.sum(sort_Lamda)

        #loadings of first two principal components
        loadings = sort_Q[:, :2]
    
    elif method=='svd':
        print('SVD is used...')
        
        #Center the data
        X_centered = (data - np.mean(data, axis=0)) / np.std(data, axis=0) #axis=0 choosing all features     

        # SVD
        U, Sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        #Keep only the top eigenvectors (select principal components)
        V_top = Vt[:keep_dims, :] # shape:(keep_dims, features)
        
        #Project the data
        X_projected = np.dot(X_centered, V_top.T)
    
        #Compute explained variance
        variances = Sigma**2 / (X_centered.shape[0]-1) # Calculate individual variance
        total_variance = np.sum(np.var(X_centered, axis=0))  # Sum of variances of original features
        explained_variance = variances / total_variance

        #loadings of first two principal components
        loadings = Vt[:2, :].T #transpose beacause PCs need to be columns
    

    elif method=='em':
        print('EM algorithm is used...')
    
        #Substract the mean from X
        X_centered = (data - np.mean(data, axis=0)).T # shape: D x N
        n_features, n_samples = X_centered.shape
    
        #Initialize a random matrix W and Psi
        W = np.random.randn(n_features, keep_dims)
    
        #Repeat EM steps
        max_iter = 1000
        tolerance = 1e-6
        for i in range(max_iter):
        
            #E-step(Estimate Omega)
            WT_W = np.dot(W.T, W)         # Equivalent to W^T x W
            WT_X = np.dot(W.T, X_centered)        # Equivalent to W^T x X
            Omega = np.linalg.solve(WT_W, WT_X)
        
            #M-step(Calculate the new parameters - Update W)
            numerator = np.dot(X_centered, Omega.T)
            denominator = np.dot(Omega, Omega.T)
            try:
                W_new = np.dot(numerator, np.linalg.inv(denominator))
            except np.linalg.LinAlgError:
                print(f"Matrix is singular at iteration {i}")
                break
    
            #Check for govergence
            if np.linalg.norm(W_new - W) < tolerance:
                print(f"Converged at iteration {i}")
                break
            W = W_new
                           
        #Project the data
        X_projected = Omega.T

        #Compute explained variance
        total_variance = np.sum(np.var(X_centered.T, axis=0, ddof=1))  # Sum of variances of original features
        projected_variance = np.var(X_projected, axis=0, ddof=1)
        explained_variance = projected_variance / total_variance

        #loadings of first two principal components
        loadings = W[:, :2]
    
    else:
        raise ValueError('Input Error. Method should be `eig`, `svd` or `em`')
        
     #Calculate cumulative variance  
    cumulative_variance = np.cumsum(explained_variance)
    
    #Finding the dimensions that explain the 80% of the total variance
    counter=0
    for i in cumulative_variance:
        counter+=1
        if i>0.8:
            print(f'For 80% of the total variance we must keep {counter} dimensions')
            break
    
    return (X_projected, loadings, explained_variance, cumulative_variance)

#Choose among 'eig', 'SVD', 'EM'
X_projected, loadings, exp_variance, cum_variance = custom_pca(X, keep_dims=2, method='eig')

#X_projected, loadings, exp_variance, cum_variance = custom_pca(X, keep_dims=2, method='svd')

#X_projected, loadings, exp_variance, cum_variance = custom_pca(X, keep_dims=2, method='em')



#plot cumulative variance explained (scree plot)
import matplotlib.pyplot as plt
import numpy as np

components = np.arange(1, len(cum_variance)+1)

plt.plot(components, cum_variance, marker='o')
plt.axhline(0.8, color='r', linestyle='--', label='80% threshold')
plt.title("Cumulative Explained Variance 'Eig'") #choose for Eig decomposition
#plt.title("Cumulative Explained Variance 'SVD'") #choose for SVD
#plt.title("Cumulative Explained Variance 'EM'") #choose for EM algorithm
plt.xlabel("Component Number")
plt.ylabel("Cumulative Variance")
plt.grid(True)
plt.legend()
plt.show()

# Loadings in bar plots
n_features = loadings.shape[0]
feature_names = [f"x{i+1}" for i in range(n_features)]

plt.bar(feature_names, loadings[:, 0])
#plt.title("Loadings for 'eig' PC 1")
#plt.title("Loadings for 'SVD' PC 1")
plt.title("Loadings for 'EM' PC 1")
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Feature")
plt.ylabel("Loading")
plt.xticks(rotation=45)
plt.show()

plt.bar(feature_names, loadings[:, 1])
#plt.title("Loadings for 'eig' PC 2")
#plt.title("Loadings for 'SVD' PC 2")
plt.title("Loadings for 'EM' PC 2")
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Feature")
plt.ylabel("Loading")
plt.xticks(rotation=45)
plt.show()

# Plot the first two principal components (columns 0 and 1)

plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Data projected via 'Eig' onto first 2 principal components")
#plt.title("Data projected via 'SVD' onto first 2 principal components")
#plt.title("Data projected via 'EM' onto first 2 principal components")
plt.grid(True)
plt.show()