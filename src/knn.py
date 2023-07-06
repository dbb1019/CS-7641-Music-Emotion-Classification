import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import os
from utils import preprocess_data, calculate_metrics, plot_supervised

X_train, X_test, y_train, y_test = preprocess_data(test_size=0.2, random_state=1234)
from sklearn.decomposition import PCA


# train model
emotion_accuracy_list = []
emotion_precision_list = []
emotion_recall_list = []
emotion_f1_score_list = []
subemotion_accuracy_list = []
subemotion_precision_list = []
subemotion_recall_list = []
subemotion_f1_score_list = []
mse_score_list = []
r2_score_list = []

k_range = range(1, 100)

for k in k_range:
    model = KNeighborsRegressor(n_neighbors=k, metric='euclidean')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    (
        emotion_accuracy,
        emotion_precision,
        emotion_recall,
        emotion_f1_score,
        subemotion_accuracy,
        subemotion_precision,
        subemotion_recall,
        subemotion_f1_score,
        r2,
        mse,
    ) = calculate_metrics(y_pred, y_test)

    emotion_accuracy_list.append(emotion_accuracy)
    emotion_precision_list.append(emotion_precision)
    emotion_recall_list.append(emotion_recall)
    emotion_f1_score_list.append(emotion_f1_score)
    subemotion_accuracy_list.append(subemotion_accuracy)
    subemotion_precision_list.append(subemotion_precision)
    subemotion_recall_list.append(subemotion_recall)
    subemotion_f1_score_list.append(subemotion_f1_score)
    r2_score_list.append(r2)
    mse_score_list.append(mse)

# plot results
directory = "../results/knn"

plot_supervised(
    "Performance of KNN with different number of PCA components",
    "Number of k",
    k_range,
    directory,
    "knn_performance_k.png",
    emotion_accuracy_list,
    emotion_precision_list,
    emotion_recall_list,
    emotion_f1_score_list,
    subemotion_accuracy_list,
    subemotion_precision_list,
    subemotion_recall_list,
    subemotion_f1_score_list,
    mse_score_list,
    r2_score_list,
)

# use K = 10 and test different number of components
emotion_accuracy_list = []
emotion_precision_list = []
emotion_recall_list = []
emotion_f1_score_list = []
subemotion_accuracy_list = []
subemotion_precision_list = []
subemotion_recall_list = []
subemotion_f1_score_list = []
mse_score_list = []
r2_score_list = []

n_components_range = range(2, X_train.shape[1] + 1, 10)
for n_components in n_components_range:
    print("n_components:", n_components)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model = KNeighborsRegressor(n_neighbors=10, metric='euclidean')
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)

    (
        emotion_accuracy,
        emotion_precision,
        emotion_recall,
        emotion_f1_score,
        subemotion_accuracy,
        subemotion_precision,
        subemotion_recall,
        subemotion_f1_score,
        r2,
        mse,
    ) = calculate_metrics(y_pred, y_test)

    emotion_accuracy_list.append(emotion_accuracy)
    emotion_precision_list.append(emotion_precision)
    emotion_recall_list.append(emotion_recall)
    emotion_f1_score_list.append(emotion_f1_score)
    subemotion_accuracy_list.append(subemotion_accuracy)
    subemotion_precision_list.append(subemotion_precision)
    subemotion_recall_list.append(subemotion_recall)
    subemotion_f1_score_list.append(subemotion_f1_score)
    r2_score_list.append(r2)
    mse_score_list.append(mse)

plot_supervised(
    "Performance of KNN with different number of PCA components",
    "Number of PCA Components",
    n_components_range,
    directory,
    "knn_performance_pca.png",
    emotion_accuracy_list,
    emotion_precision_list,
    emotion_recall_list,
    emotion_f1_score_list,
    subemotion_accuracy_list,
    subemotion_precision_list,
    subemotion_recall_list,
    subemotion_f1_score_list,
    mse_score_list,
    r2_score_list,
)
    