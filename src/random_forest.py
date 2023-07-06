from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from utils import preprocess_data, calculate_metrics, plot_supervised
from sklearn.decomposition import PCA


X_train, X_test, y_train, y_test = preprocess_data(test_size=0.2, random_state=1234)

# train model
n_estimators_range = [10, 50, 100, 250, 500, 750]
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
for n_estimators in n_estimators_range:
    print("n_estimators:", n_estimators)
    base_model = RandomForestRegressor(n_estimators=n_estimators, random_state=1234)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    # evaluate model
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


# plot mse results
directory = "../results/random_forest"

plot_supervised(
    "Performance of Random Forest with different number of estimators",
    "Number of estimators",
    n_estimators_range,
    directory,
    "rf_performance_n_estimators.png",
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

n_components_range = range(2, X_train.shape[1] + 1, 10)
for n_components in n_components_range:
    print("n_components:", n_components)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    base_model = RandomForestRegressor(n_estimators=100, random_state=1234)
    model = MultiOutputRegressor(base_model)
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
    "Performance of Random Forest with different number of PCA components",
    "Number of PCA Components",
    n_components_range,
    directory,
    "rf_performance_pca.png",
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
