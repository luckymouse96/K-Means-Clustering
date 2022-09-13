from sklearn.cluster import KMeans

def doKMeans(data, num_clusters=0):
    useful_features = ['TowerLat', 'TowerLon']
    x_train = data[useful_features].copy()
    model = KMeans(n_clusters=num_clusters, random_state=0).fit(x_train)
    return model
def clusterInfo(model):
    print("Cluster Analysis Inertia: ", model.inertia_)
    print('------------------------------------------')
    
    for i in range(len(model.cluster_centers_)):
        print("\n  Cluster (Vị trí có thể là nhà hoặc nơi làm việc) ", i)
        print("    Tọa độ ", model.cluster_centers_[i])
        print("    #Số lượng ", (model.labels_==i).sum()) # NumPy Power
def clusterWithFewestSamples(model):
    # Ensure there's at least on cluster...
    minSamples = len(model.labels_)
    minCluster = 0
    
    for i in range(len(model.cluster_centers_)):
        if minSamples > (model.labels_==i).sum():
            minCluster = i
            minSamples = (model.labels_==i).sum()

    print("\n  Cluster With Fewest Samples: ", minCluster)
    return (model.labels_==minCluster)
