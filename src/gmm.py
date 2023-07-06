#!/usr/bin/env python
# coding: utf-8

# In[12]:


from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# load data and labels

data1 = pd.read_csv(
    "./archive/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
)
data2 = pd.read_csv(
    "./archive/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv"
)
data = pd.concat([data1, data2])

# print(data.head())

# print(data)
labels = data.iloc[:, 1:5]

# print(labels)
# print(labels.head())

features = []
for song_id in data["song_id"]:
    song_features = pd.read_csv(f"./archive/features/features/{song_id}.csv", sep=";")
    song_features = song_features.iloc[:, 1:]
#     print(song_features)
    song_features_mean = song_features.mean(axis=0)
    features.append(song_features_mean.values)
    
features = np.array(features)


# print(features_pca)



# In[5]:


# features = features_pca
# normalize features


# In[35]:


# print(X_train.shape)

for j in range(5):
    accuracy_array = []
    silhouette_array = []
    pca = PCA(n_components=j+2)

    # Fit PCA on the data
    features_pca = pca.fit_transform(features)
    scaler = StandardScaler()
    features_pca = scaler.fit_transform(features_pca)

    # data split
    X_train, X_test, y_train, y_test = train_test_split(
        features_pca, labels, test_size=0.2, random_state=1234
    )
    for i in range(10):
        components = i+2
        gmm_x = GaussianMixture(n_components=components)
        gmm_x.fit(X_train)
        labels_x = gmm_x.predict(X_test)



        gmm_y = GaussianMixture(n_components=components)
        gmm_y.fit(y_train)
        labels_y = gmm_y.predict(y_test)

    #     print (labels_y == labels_x)
        silhouette_avg = silhouette_score(X_test, labels_x)
        
        print('For ',components,' components, ','the accuracy rate is: ',np.mean(labels_y == labels_x)*100,"%", ' \n         The Silhouette Scores is:', silhouette_avg)
        accuracy_array.append(np.mean(labels_y == labels_x)*100)
        silhouette_array.append(silhouette_avg)
    x_plt = range(2,12)
    # Plot accuracy rate
    label = f'PCA component = {j+2}'
    fig1 = plt.figure(1)
    plt.plot(x_plt, accuracy_array,label=label)
    plt.xlabel('GMM components')
    plt.ylabel('Accuracy rate %')
    plt.title('GMM')
    plt.ylim([0,100])
    plt.legend()
    
    # plot silhouette scores
    fig2 = plt.figure(2)
    plt.ylim([-0.2,1])
    plt.plot(x_plt, silhouette_array,label=label)
    plt.plot()
    # plt the accuracy rate vs gmm components
    plt.xlabel('GMM components')
    plt.ylabel('Silhouette scores')
    plt.title('GMM')
    plt.hold = True
    plt.legend()
# save the figures
fig1.savefig("accuracy_rate.png",dpi =300)
fig2.savefig("silhouette.png",dpi = 300)

# Show the plot
plt.show()


# In[156]:





# In[ ]:




