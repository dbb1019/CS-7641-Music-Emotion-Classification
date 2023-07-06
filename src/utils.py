import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import os


def preprocess_data(test_size=0.2, random_state=1234):
    # load data and labels
    data1 = pd.read_csv(
        "../data/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
    )
    data2 = pd.read_csv(
        "../data/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv"
    )
    data = pd.concat([data1, data2], sort=False)

    print(data.head())

    labels = data.iloc[:, [1, 3]]

    print(labels.head())

    labels = labels.values

    features = []
    for song_id in data["song_id"]:
        song_features = pd.read_csv(f"../data/features/features/{song_id}.csv", sep=";")
        song_features = song_features.iloc[:, 1:]
        song_features_mean = song_features.mean(axis=0)
        features.append(song_features_mean.values)

    features = np.array(features)
    print("features:", features.shape)

    # normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # data split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def preprocess_data_dynamic(test_size=0.2, random_state=1234):
    # load data and labels
    arousal_data = pd.read_csv(
        "../data/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv"
    ).dropna(axis="columns")

    valence_data = pd.read_csv(
        "../data/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv"
    ).dropna(axis="columns")

    # Merge arousal and valence data on 'song_id' column
    data = pd.merge(
        arousal_data, valence_data, on="song_id", suffixes=("_arousal", "_valence")
    )

    features = []

    for song_id in data["song_id"]:
        song_features = pd.read_csv(f"../data/features/features/{song_id}.csv", sep=";")
        song_features = song_features.iloc[30:90, 1:]

        # pad the dataframe with 0s if it has less than 60 rows
        if song_features.shape[0] < 60:
            pad_length = 60 - song_features.shape[0]
            padding = pd.DataFrame(
                np.zeros((pad_length, song_features.shape[1])),
                columns=song_features.columns,
            )
            song_features = pd.concat([song_features, padding], axis=0)
        features.append(song_features.values)

    features = np.stack(features)
    print("features:", features.shape)

    data = data.iloc[:, 1:]
    print(data.head())

    # data split
    X_train, X_test, y_train, y_test = train_test_split(
        features, data.values, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def get_emotion_type(arousal, valence):
    emotion_map = {"Joy": 0, "Fear": 1, "Contentment": 2, "Sadness": 3}
    subemotion_map = {
        "Elation": 0,
        "Excitement": 1,
        "Enthusiasm": 2,
        "Terror": 3,
        "Anger": 4,
        "Hostility": 5,
        "Serenity": 6,
        "Relaxation": 7,
        "Calmness": 8,
        "Melancholy": 9,
        "Boredom": 10,
        "Disgust": 11,
    }
    if arousal >= 5 and valence >= 5:
        emotion = "Joy"
    elif arousal >= 5 and valence < 5:
        emotion = "Fear"
    elif arousal < 5 and valence >= 5:
        emotion = "Contentment"
    else:
        emotion = "Sadness"

    if emotion == "Joy":
        if valence >= 7.5:
            subemotion = "Elation"
        elif valence >= 5 and valence < 7.5:
            subemotion = "Excitement"
        else:
            subemotion = "Enthusiasm"
    elif emotion == "Fear":
        if valence >= 7.5:
            subemotion = "Terror"
        elif valence >= 5 and valence < 7.5:
            subemotion = "Anger"
        else:
            subemotion = "Hostility"
    elif emotion == "Contentment":
        if valence >= 7.5:
            subemotion = "Serenity"
        elif valence >= 5 and valence < 7.5:
            subemotion = "Relaxation"
        else:
            subemotion = "Calmness"
    else:
        if valence >= 7.5:
            subemotion = "Melancholy"
        elif valence >= 5 and valence < 7.5:
            subemotion = "Boredom"
        else:
            subemotion = "Disgust"

    # Convert emotions to integer values using the mapping dictionaries
    emotion = emotion_map[emotion]
    subemotion = subemotion_map[subemotion]

    return emotion, subemotion


def calculate_metrics(y_pred, y_test):
    arousal_valence_pred = np.column_stack((y_pred, y_test))
    print("arousal_valence_pred:", arousal_valence_pred.shape)
    emotion_pred_list = []
    subemotion_pred_list = []
    emotion_gt_list = []
    subemotion_gt_list = []
    for arousal, valence, arousal_gt, valence_gt in arousal_valence_pred:
        emotion_pred, subemotion_pred = get_emotion_type(arousal, valence)
        emotion_gt, subemotion_gt = get_emotion_type(arousal_gt, valence_gt)
        emotion_pred_list.append(emotion_pred)
        subemotion_pred_list.append(subemotion_pred)
        emotion_gt_list.append(emotion_gt)
        subemotion_gt_list.append(subemotion_gt)

    emotion_accuracy = accuracy_score(emotion_gt_list, emotion_pred_list)
    print("Classification Accuracy (emotion):", round(emotion_accuracy, 3))

    subemotion_accuracy = accuracy_score(subemotion_gt_list, subemotion_pred_list)
    print("Classification Accuracy (subemotion):", round(subemotion_accuracy, 3))

    (
        emotion_precision,
        emotion_recall,
        emotion_f1_score,
        _,
    ) = precision_recall_fscore_support(
        emotion_gt_list,
        emotion_pred_list,
        average="weighted",
        labels=np.unique(emotion_pred_list),
    )
    print("Precision (emotion):", round(emotion_precision, 3))
    print("Recall (emotion):", round(emotion_recall, 3))
    print("F1 score (emotion):", round(emotion_f1_score, 3))

    (
        subemotion_precision,
        subemotion_recall,
        subemotion_f1_score,
        _,
    ) = precision_recall_fscore_support(
        subemotion_gt_list,
        subemotion_pred_list,
        average="weighted",
        labels=np.unique(subemotion_pred_list),
    )
    print("Precision (subemotion):", round(subemotion_precision, 3))
    print("Recall (subemotion):", round(subemotion_recall, 3))
    print("F1 score (subemotion):", round(subemotion_f1_score, 3))

    r2 = r2_score(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    print("Mean Squared Error:", round(mse, 3))
    print("R2 Score:", round(r2, 3))
    return (
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
    )


def plot_supervised(
    title,
    xlabel,
    x_range,
    output_folder,
    filename,
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
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # colors = ["r", "g", "b", "c", "m", "y", "k", "darkorange", "purple"]
    colors = ["r", "g", "b", "c", "m", "y"]
    # labels = [
    #     "Emotion Accuracy",
    #     "Emotion Precision",
    #     "Emotion Recall",
    #     "Emotion F1-Score",
    #     "Sub-Emotion Accuracy",
    #     "Sub-Emotion Precision",
    #     "Sub-Emotion Recall",
    #     "Sub-Emotion F1-Score",
    #     "MSE Score",
    #     "R2 Score",
    # ]
    labels = [
        "Emotion Accuracy",
        "Emotion F1-Score",
        "Sub-Emotion Accuracy",
        "Sub-Emotion F1-Score",
        "MSE Score",
        "R2 Score",
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, data_list in enumerate(
        [
            emotion_accuracy_list,
            # emotion_precision_list,
            # emotion_recall_list,
            emotion_f1_score_list,
            subemotion_accuracy_list,
            # subemotion_precision_list,
            # subemotion_recall_list,
            subemotion_f1_score_list,
            mse_score_list,
            r2_score_list,
        ]
    ):
        ax.plot(
            x_range,
            data_list,
            marker="",
            color=colors[i % len(colors)],
            label=labels[i],
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")

    plt.title(title)
    plt.legend()

    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)
    plt.cla()

    # find the index of the largest value in the accuracy list
    max_acc_idx = r2_score_list.index(max(r2_score_list))

    # output the evaluation metrics for that index
    print("For the main emotion prediction:")
    print("Accuracy:", round(emotion_accuracy_list[max_acc_idx], 3))
    print("Precision:", round(emotion_precision_list[max_acc_idx], 3))
    print("Recall:", round(emotion_recall_list[max_acc_idx], 3))
    print("F1-score:", round(emotion_f1_score_list[max_acc_idx], 3))
    print("For the subemotion prediction:")
    print("Accuracy:", round(subemotion_accuracy_list[max_acc_idx], 3))
    print("Precision:", round(subemotion_precision_list[max_acc_idx], 3))
    print("Recall:", round(subemotion_recall_list[max_acc_idx], 3))
    print("F1-score:", round(subemotion_f1_score_list[max_acc_idx], 3))
    print("MSE score:", round(mse_score_list[max_acc_idx], 3))
    print("R2 score:", round(r2_score_list[max_acc_idx], 3))
