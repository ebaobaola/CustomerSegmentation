
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
get_ipython().magic(u'matplotlib inline')

sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")


# Below is the outline for this notebook:
# * <a href='#load'>Load the Data</a>
# * <a href='#first'>First Iteration of E Step and M Step</a>
# * <a href='#second'>Second Iteration of E Step and M Step</a>
# * <a href='#third'>Third Iteration of E Step and M Step</a>
# * <a href='#general'>The Iterative Process</a>
# * <a href='#business'>Additional Analysis of Clusters and Business Findings</a>
# * <a href='#other'>Other Applications</a>
# * <a href='#sources'>Sources</a>

# # Load the Data <a id='load'></a>

# The synthesized dataset contains both information on marketing newsletters/e-mail campaigns (e-mail offers sent) and transaction level data from customers (which offer customers responded to and what they bought).
# 
# The marketing campaign focused on two aspects of an insurance product: amount of premium savings the customer can receive if he/she chooses the product, and the closeness and accessibility to an insurance agent in the same neighborhood.  The level of focuses on each of the aspect is ranked from 0 to 10.

# In[4]:

df_offers = pd.read_excel("./Insurance Product Offering.xlsx", sheetname=0)
df_offers.columns = ["offer_id", "campaign", "product_type", "level_emphasis_premium_savings",                      "level_emphasis_neighborhood_agent"]
df_offers.head()


# The responses tab in the spreadsheet shows which offer the customer responded to or expressed interest in.  In addition to the customer_name and offer_id columns that come with the spreadsheet, we also added a column 'n' that is always 1.

# In[7]:

df_responses = pd.read_excel("./Insurance Product Offering.xlsx", sheetname=1)
df_responses.columns = ["customer_name", "offer_id"]
df_responses['n'] = 1
df_responses.head()


# In[8]:

# join the offers and responses tables
df = pd.merge(df_offers, df_responses)
df.head()


# In[9]:

# create a "pivot table" which will give us the number of times each customer responded to a given offer
matrix = df.pivot_table(index=['customer_name'], columns=['offer_id'], values='n')
print matrix


# In[10]:

# fill NA values with 0 and make the index into a column
matrix = matrix.fillna(0).reset_index()
print matrix


# In[11]:

# save a list of the 0/1 columns (offer id). we'll use these a bit later
x_cols = matrix.columns[1:]
print x_cols


# In[12]:

data = matrix[x_cols]


# In[13]:

data = data.as_matrix()


# In[14]:

print data


# # Below is code for Gaussian Mixture Models with Expectation-Maximization:

# In[15]:

def mv_normal_log_pdf(X, mu, Sig):
    return (-0.5*np.linalg.slogdet(2*np.pi*Sig)[1]
     - 0.5*np.sum((X-mu)*(np.linalg.inv(Sig).dot((X-mu).T)).T,axis=1))


# In[16]:

class GaussianMixture:
    def init_parameters(self, X, P, k, reg):
        """ Initialize the parameters of means, covariances, and frequency counts. 
            Args: 
                X (numpy 2D matrix) : data matrix, each row is an example
                P (numpy 2D matrix) : Random permutation vector
                k (float) : number of clusters
                reg (float) : regularization parameter for the covariance matrix
            Returns: 
                mus (numpy 2D matrix) : matrix of initialzied means, chosen randomly by selection the first k elements of P
                Sigmas (list) : list of 2D covariance matrices corresponding to each cluster
                phi (numpy 1D vector) : vector of initialized frequencies
        """
        # initlize the centers mu. each row of the matrix is a center
        # we use the first k random indices of the permutation as centers
        mu_index = P[:k]  
        Mu = X[mu_index,:]
        
        #Covariance matrix should be intialized as the sample covariance of the 
        #entire data matrix plus regularization using the unbiased estimater. 
        sigma = np.cov(X.T + reg)
        sigmas =[sigma] * k  #list of 2D covariance matrices corresponding to each cluster/component
        
        # start with equal probability for all the clusters/gaussian components
        phi = np.empty(k)  
        phi.fill(1.0/float(k))
        return Mu, sigmas, phi
        
        

    def Estep(self, X, mus, Sigmas, phi):
        """ Perform an E step and return the resulting probabilities. 
            Args: 
                X (numpy 2D matrix) : data matrix, each row is an example
                mus (numpy 2D matrix) : matrix of initialzied means, chosen randomly by selection the first k elements of P
                Sigmas (list) : list of 2D covariance matrices corresponding to each cluster
                phi (numpy 1D vector) : vector of initialized frequencies
            Returns: 
                (numpy 2D matrix) : matrix of probabilities, where the i,jth entry corresponds to the probability of the
                                    ith element being in the jth cluster. 
        """
        # matrix of probabilities. where the i,jth entry corresponds to the probability of the ith element being in
        # jth cluster/component
        mx_prob = np.zeros((X.shape[0],len(Sigmas)))
        for j in range(mx_prob.shape[1]):
            # multivariate normal log pdf function
            mx_prob[:,j] = mv_normal_log_pdf(X, mus[j, :], Sigmas[j]) + np.log(phi[j])
        #log trick: dividing by a very large negative number can cause numerical problems, since everything is 0
        # so we add a small number to numerator and denominator before division
        # it is common to use the row max so at least one element in the resulting probability vector is 1
        log_trick = np.max(mx_prob,axis=1)
        mx_prob = np.exp(mx_prob-log_trick[:, np.newaxis])
        mx_prob = mx_prob/np.sum(mx_prob, axis=1)[:,None]
        
        return mx_prob
        

    def Mstep(self, ps, X, reg):
        """  
            Args: 
                ps (numpy 2D matrix) : matrix of probabilities, where the i,jth entry corresponds to the probability of the
                                       ith element being in the jth cluster. 
                X (numpy 2D matrix) : data matrix, each row is an example
                reg (float) : regularization parameter for the covariance matrix
            Returns: 
                (mus, Sigmas, phi) : 3 tuple of matrix of initialzied means, chosen randomly by selection the first 
                                     k elements of P, a list of 2D covariance matrices corresponding to each cluster, 
                                     and a vector of initialized frequencies
        """
        # sum up the probability of being in jth cluster (column sum), divided by number of rows in ps (num of records)
        phi = np.sum(ps, axis = 0)/float(ps.shape[0])
        # ps transpose dot X, divided by column sum of ps
        mus = ps.T.dot(X)/ np.sum(ps, axis=0)[:, None]

        Sigmas = []
        for k in range(ps.shape[1]):  # for each cluster
            step0 = X - mus[k,:]
            step1 = step0.T*(ps[:, k]) #step0 is m by n, step0.T is n by m, ps[:,k] is m by 1 so its matrix vector multi
                                       # step1 is n by m
            step2 = step1.dot(step0)   #step2 is n by n
            step3 = step2/float(np.sum(ps[:,k])) #n by n
            # add covariance matrix regulaization
            sigma = step3 + np.eye(step3.shape[1])*reg
            #append to the list of covariance matrices
            Sigmas.append(sigma)
        return (mus, Sigmas, phi)
        
        
        
        
            
    
    def train(self, X, mus, Sigmas, phi, niters = 5, reg=1e-4):
        """ Train the model using the EM algorithm for a number of iterations. 
            Args: 
                X (numpy 2D matrix) : data matrix, each row is an example
                mus (numpy 2D matrix) : matrix of initialzied means, chosen randomly by selection the first k elements of P
                Sigmas (list) : list of 2D covariance matrices corresponding to each cluster
                phi (numpy 1D vector) : vector of initialized frequencies
                niters (int) : number of EM iterations to run
            Returns: 
                (mus, Sigmas, phi) : 3 tuple of matrix of initialzied means, chosen randomly by selection the first 
                                     k elements of P, a list of 2D covariance matrices corresponding to each cluster, 
                                     and a vector of initialized frequencies 
        
        """
        for i in range(niters):
            mx_prob = self.Estep(X, mus, Sigmas, phi)
            mus, Sigmas, phi = self.Mstep(mx_prob, X, reg)
        return (mus, Sigmas, phi)


# # First Iteration of E Step and M Step <a id='first'></a>

# In[22]:

k = 4
P = np.arange(data.shape[0])
reg = 1e-4
GM = GaussianMixture()
mus, Sigmas, phi = GM.init_parameters(data, P, k, reg)
ps = GM.Estep(data, mus, Sigmas, phi)
mus, Sigmas, phi = GM.Mstep(ps, data, reg)


# In[23]:

print ps.shape
print ps


# In[24]:

cluster_1 = np.argmax(ps, axis = 1) #in the actual function, a data point contributes to recomputing the parameters of
                                    # every single Guassian component it can potentially belong to


# In[25]:

matrix['cluster_1'] = cluster_1


# # To visualize the cluster, we performed a dimension reduction:

# In[26]:

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
matrix['x'] = pca.fit_transform(matrix[x_cols])[:,0]
matrix['y'] = pca.fit_transform(matrix[x_cols])[:,1]
matrix = matrix.reset_index()

customer_clusters = matrix[['customer_name', 'cluster_1', 'x', 'y']]
print customer_clusters


# In[27]:

customer_clusters.cluster_1.value_counts()


# In[29]:

#https://chrisalbon.com/python/seaborn_scatterplot.html

sns.lmplot('x', 'y',
           data=customer_clusters,
           fit_reg=False,
           hue='cluster_1',  
           scatter_kws={"marker": "D",
                        "s": 100})
plt.title('Customer Segmentation')
plt.xlabel('x')
plt.ylabel('y')


# # Second Iteration of E Step and M Step <a id='second'></a>

# In[30]:

ps = GM.Estep(data, mus, Sigmas, phi)
mus, Sigmas, phi = GM.Mstep(ps, data, reg)


# In[31]:

cluster_2 = np.argmax(ps, axis = 1)


# In[32]:

matrix['cluster_2'] = cluster_2
customer_clusters = matrix[['customer_name', 'cluster_1','cluster_2', 'x', 'y']]
print customer_clusters


# In[33]:

customer_clusters.cluster_2.value_counts()


# In[34]:

sns.lmplot('x', 'y',
           data=customer_clusters,
           fit_reg=False,
           hue='cluster_2',  
           scatter_kws={"marker": "D",
                        "s": 100})
plt.title('Customer Segmentation')
plt.xlabel('x')
plt.ylabel('y')


# # Third Iteration of E Step and M Step <a id='third'></a>

# In[35]:

ps = GM.Estep(data, mus, Sigmas, phi)
mus, Sigmas, phi = GM.Mstep(ps, data, reg)
cluster_3 = np.argmax(ps, axis = 1)
matrix['cluster_3'] = cluster_3
customer_clusters = matrix[['customer_name', 'cluster_1','cluster_2', 'cluster_3','x', 'y']]
print customer_clusters.cluster_3.value_counts()


# In[36]:

customer_clusters.head()


# In[37]:

sns.lmplot('x', 'y',
           data=customer_clusters,
           fit_reg=False,
           hue='cluster_3',  
           scatter_kws={"marker": "D",
                        "s": 100})
plt.title('Customer Segmentation')
plt.xlabel('x')
plt.ylabel('y')


# # The Iterative Process <a id='general'></a>

# In[38]:

mus, Sigmas, phi = GM.train(data, mus, Sigmas, phi, niters = 150, reg=1e-4)
ps = GM.Estep(data, mus, Sigmas, phi)
cluster_final = np.argmax(ps, axis = 1)
matrix['cluster_final'] = cluster_final
customer_clusters = matrix[['customer_name', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_final','x', 'y']]
print customer_clusters.cluster_final.value_counts()
customer_clusters.head()


# In[39]:

sns.lmplot('x', 'y',
           data=customer_clusters,
           fit_reg=False,
           hue='cluster_final',  
           scatter_kws={"marker": "D",
                        "s": 100})
plt.title('Customer Segmentation')
plt.xlabel('x')
plt.ylabel('y')


# # Additional Analysis of Clusters and Business Findings <a id='business'></a>

# In[40]:

df = pd.merge(df_transactions, customer_clusters)
df = pd.merge(df_offers, df)
df.head()


# In[41]:

df['is_0'] = df.cluster_final==0
print df.groupby("is_0").product_type.value_counts()


# In[43]:

print df.groupby("is_0")[['level_emphasis_premium_savings', 'level_emphasis_neighborhood_agent']].mean()
print df.groupby("is_0")[['level_emphasis_premium_savings', 'level_emphasis_neighborhood_agent']].median() #group 0 likes offers with smaller quantities


# In[45]:

df['is_1'] = df.cluster_final==1
print df.groupby("is_1").product_type.value_counts()  


# In[46]:

print df.groupby("is_1")[['level_emphasis_premium_savings', 'level_emphasis_neighborhood_agent']].mean()
print df.groupby("is_1")[['level_emphasis_premium_savings', 'level_emphasis_neighborhood_agent']].median() #nothing too significant, one order must have skewed the mean


# In[47]:

df['is_2'] = df.cluster_final==2
print df.groupby("is_2").product_type.value_counts() 


# In[48]:

print df.groupby("is_2")[['level_emphasis_premium_savings', 'level_emphasis_neighborhood_agent']].mean()
print df.groupby("is_2")[['level_emphasis_premium_savings', 'level_emphasis_neighborhood_agent']].median() #nothing too significant, care about discount a little less


# In[49]:

df['is_3'] = df.cluster_final==3
print df.groupby("is_3").product_type.value_counts() 


# In[50]:

print df.groupby("is_3")[['level_emphasis_premium_savings', 'level_emphasis_neighborhood_agent']].mean()
print df.groupby("is_3")[['level_emphasis_premium_savings', 'level_emphasis_neighborhood_agent']].median() #buys larger quantities


# # Applications for Other Domains<a id='other'></a>

# Market segmentation for the financial sector
# 
# Anomaly detection of transactions
# 
# News clustering (TF-IDF)
# 
# Medical image segmentation
# 
# etc.
# 

# # Sources<a id='sources'></a>

# http://blog.yhat.com/posts/customer-segmentation-using-python.html
# 
# http://ieeexplore.ieee.org/document/1633722/?reload=true
# 
# http://www.nature.com/nbt/journal/v26/n8/full/nbt1406.html#f1
# 
# https://www.researchgate.net/post/How_does_GMM_work_in_principle  
# 
# http://datasciencecourse.org/anomaly_detection.pdf 
# 
# http://cs229.stanford.edu/notes/cs229-notes8.pdf 
# 
