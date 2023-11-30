## Code for A5 Assignment

# import dependencies
import numpy as np
import matplotlib.pyplot as plt


# Part A - kNN Classifier

# load feature data, genre labels and feature names from files
data = np.loadtxt('A5/data/data.txt')
labels = np.loadtxt('A5/data/labels.txt')

# load feature names text file as a dictionary and ignore the first line
feature_dict = {}
with open('A5/data/feature_dict.txt') as f:
    for line in f:
        if line.startswith('//'):
            continue
        line = line.rstrip()
        key, val = line.split(sep=', ', maxsplit=2)
        feature_dict[val] = key

# load genre labels text file as a dictionary and ignore the first line
genre_dict = {}
with open('A5/data/label_dict.txt') as f:
    for line in f:
        if line.startswith('//'):
            continue
        line = line.rstrip()
        key, val = line.split(sep=', ', maxsplit=2)
        genre_dict[val] = key


                                                                            # Part A.1 - k-NN classifier

def knearestneighbor(test_data, train_data, train_label, k):
    # knn directly does inference on test data since only distance between test and train data is needed
    #  loop through each test data point and calculate distance to each train data point, then sort distances to find the k nearest neighbors for each test data point
    # calculate the average label of the k nearest neighbors and assign that label to the test data point

    Npoints = max(np.shape(train_data)) # number of training data points
    # shuffle the training data and labels in the same order
    shuffle = np.random.permutation(Npoints)
    train_data = train_data[...,shuffle]
    train_label = train_label[shuffle]
    Nt = max(np.shape(test_data)) # number of test data points

    est_class = []
    dist = []

    for i in range(Nt):  
        for j in range(Npoints):
            dist.append(np.linalg.norm(test_data[...,i]-train_data[...,j]))
        dist = np.array(dist)
        dist_sort = np.argsort(dist)
        dist_sort = dist_sort[:k]
        est_class.append(np.mean(train_label[dist_sort]))
        dist = []

    return est_class

                                                                                # Part A.2 - In report 



                                                                                # Part B - Implement cross-validation
def cross_validate(data, gt_labels, k, num_folds):
    # first shuffle the data and labels in the same order
    Npoints = max(np.shape(data))
    shuffle = np.random.permutation(Npoints)
    data = data[...,shuffle]
    gt_labels = gt_labels[shuffle]
    N = num_folds

    # since the data is shuffled, split the data in train and test by selecting every N-1 data points as training data and the remaining data points as test data
    fold_size = np.floor(Npoints/N).astype(int)

    # now loop through each fold and use it as the test data while the rest of the folds are used as training data
    fold_accuracies = []
    # get total number of unique labels from the ground truth labels
    Nlabels = len(np.unique(gt_labels))
    # create a confusion matrix for test data
    conf_mat = np.zeros((Nlabels,Nlabels))
    for i in range(num_folds):
        test_ind = np.arange(i*fold_size,(i+1)*fold_size)
        non_test_ind = np.concatenate((np.arange(0,i*fold_size),np.arange((i+1)*fold_size,Npoints)))
        test_data = data[...,test_ind]
        test_labels = gt_labels[test_ind]
        train_data = data[...,non_test_ind]
        train_labels = gt_labels[non_test_ind]
        est_labels = knearestneighbor(test_data, train_data, train_labels, k)
        fold_accuracies.append(np.sum(est_labels == test_labels)/len(test_labels))
        # update the confusion matrix
        for j in range(len(test_labels)):
            conf_mat[int(test_labels[j])-1,int(est_labels[j])-1] += 1   
    avg_accuracy = np.mean(fold_accuracies)


    return avg_accuracy, fold_accuracies, conf_mat


                                                                                    # Part B.2 - find best features 

# first find the best feature by running cross-validation for each feature
def find_best_features(data, labels, k, num_folds):
    Nfeatures = np.shape(data)[0]
    best_feature = np.zeros(Nfeatures)
    best_accuracy = 0
    for i in range(Nfeatures):
        avg_accuracy, fold_accuracies, conf_mat = cross_validate(data[i,:], labels, k, num_folds)
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_feature = i

    feature_index = best_feature
    print("best feature is: ", feature_dict[str(int(best_feature)+1)], best_feature)
    return feature_index

# report best feature by running on training data, k = 3, num_folds = 3
feature_index = find_best_features(data, labels, 3, 3) # function prints output directly - added to report




                                                                                    # Part C - Feature Selection
def select_features(data, labels, k, num_folds):
    # find accuracy for different number of features
    # first find accuracy for all features, and sort the features by accuracy, then find accuracy for the best feature, then the best 2 features, etc.
    Nfeatures = np.shape(data)[0]
    accuracy = []
    sel_feature_ind = []
    for i in range(Nfeatures):
        avg_accuracy, fold_accuracies, conf_mat = cross_validate(data[i,:], labels, k, num_folds)
        accuracy.append(avg_accuracy)
        sel_feature_ind.append(i)
    accuracy = np.array(accuracy)
    sel_feature_ind = np.array(sel_feature_ind)
    sort_ind = np.argsort(accuracy)
    sort_ind = sort_ind[::-1]
    accuracy = accuracy[sort_ind]
    sel_feature_ind = sel_feature_ind[sort_ind]
    
    # now loop through each feature and find the accuracy for that feature and all the features before it
    for i in range(Nfeatures):
        avg_accuracy, fold_accuracies, conf_mat = cross_validate(data[sel_feature_ind[:i+1],:], labels, k, num_folds)
        accuracy[i] = avg_accuracy
    # plot the accuracy vs number of features
    plt.figure()
    plt.plot(np.arange(Nfeatures)+1, accuracy)
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Features')
    plt.show()


    return sel_feature_ind, accuracy

sel_feature_ind, accuracy = select_features(data, labels, 3, 3)
print(sel_feature_ind)

# find the best feature subset
best_set = sel_feature_ind[:np.argmax(accuracy)+1]
print("besy set is: ", best_set)
for i in range(len(best_set)):
    print(feature_dict[str(int(best_set[i]+1))])

# call cross_validate function with the best feature subset for varying k

def evaluate(data, labels):
    k = np.array([1, 3, 7])
    nfolds = 10
    accuracies = []
    conf_mats = []
    for i in k:
        accuracy, fold_accuracies, conf_mat = cross_validate(data[best_set,:], labels, i, nfolds)
        print("avg accuracy for k = ", i, " is: ", accuracy)
        accuracies.append(accuracy)
        conf_mats.append(conf_mat)
    
    return accuracies, conf_mats

accuracies, conf_mats = evaluate(data, labels)
print(accuracies)

# plot the confusion matrix
plt.figure()
plt.imshow(conf_mats[0], cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix for k = 1')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

plt.figure()
plt.imshow(conf_mats[1], cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix for k = 3')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

plt.figure()
plt.imshow(conf_mats[2], cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix for k = 7')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()