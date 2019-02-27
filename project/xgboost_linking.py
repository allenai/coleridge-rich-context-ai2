import xgboost as xgb
import os
import json
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
import random

def processed_docs_to_xgboost_dataset(processed_docs, input_pub_ids=None):
    """Turns the list of processed docs that the sgtb model expects into
       binary classification input for sklearn

       @param processed_docs: list of processed docs in the format that the sgtb model expects
       @param input_pub_ids: (optional) a list of publication ids for each doc, only needed at test time
    """
    feature_idx_to_name = {0: "p(e)",
                           1: "p(e|m)",
                           2: "p(m|e)",
                           3: "year_match",
                           4: "m_length_chars",
                           5: "m_length_tokens",
                           6: "max_length_sentence",
                           7: "min_length_sentence",
                           8: "is_acronym",
                           9: "background",
                           10: "methods",
                           11: "results",
                           12: "abstract",
                           13: "intro",
                           14: "introduction",
                           15: "keywords",
                           16: "objectives",
                           17: "conclusion",
                           18: "measures",
                           19: "discussion",
                           20: "method",
                           21: "references",
                           22: "contribution",
                           23: "data",
                           24: "no_section_found",
                           25: "context_word_overlap",
                           26: "score",
                           27: "is_null_candidate"}
    feature_idx_to_aggregation_method = {0: np.max,
                                         1: np.max,
                                         2: np.max,
                                         3: np.max,
                                         4: np.max,
                                         5: np.max,
                                         6: np.max,
                                         7: np.min,
                                         8: np.mean,
                                         25: np.max,
                                         26: np.max,
                                         27: np.max,}
    for i in range(9, 25):
        feature_idx_to_aggregation_method[i] = np.max

    idx_dataset_to_X = {}
    idx_dataset_to_y = {}
    dataset_ids = []
    output_pub_ids = []
    idx_dataset_to_predicted_dataset = {}
    for i, doc in enumerate(processed_docs):
        for mention_candidate in doc:
            mention = mention_candidate[0]
            entity_candidates = mention_candidate[1]
            for entity_candidate in entity_candidates:
                entity_id, label = entity_candidate[0]
                features = entity_candidate[1]
                if entity_id == "NULL":
                    continue
                key = str(i) + "_" + str(entity_id)
                idx_dataset_to_predicted_dataset[key] = entity_id
                if key in idx_dataset_to_X:
                    idx_dataset_to_X[key].append(features)
                else:
                    idx_dataset_to_X[key] = [features]
                    idx_dataset_to_y[key] = label

    X = []
    y = []
    for idx_dataset in idx_dataset_to_X:
        idx = idx_dataset.split("_")[0]
        dataset_id = idx_dataset.split("_")[1]
        np_feature_array = np.array(idx_dataset_to_X[idx_dataset])
        combined_features = [feature_idx_to_aggregation_method[i](np_feature_array[:, i]) for i in range(np_feature_array.shape[1]-1)]
        X.append(combined_features)
        y.append(idx_dataset_to_y[idx_dataset])
        dataset_ids.append(idx_dataset_to_predicted_dataset[idx_dataset])
        if input_pub_ids != None:
            output_pub_ids.append(input_pub_ids[int(idx)])

    return np.array(X), np.array(y), output_pub_ids, dataset_ids

def main():
    train_sgtb_path = os.path.abspath(os.path.join("project", "dataset_split_data", "train", "sgtb_scores.json"))
    dev_sgtb_path = os.path.abspath(os.path.join("project", "dataset_split_data", "dev", "sgtb_scores.json"))
    test_sgtb_path = os.path.abspath(os.path.join("project", "dataset_split_data", "test", "sgtb_scores.json"))

    train_processed_docs = []
    with open(train_sgtb_path, 'rb') as train_sgtb_file:
        for line in train_sgtb_file:
            train_processed_docs.append(json.loads(line.strip()))

    dev_processed_docs = []
    with open(dev_sgtb_path, 'rb') as dev_sgtb_file:
        for line in dev_sgtb_file:
            dev_processed_docs.append(json.loads(line.strip()))

    test_processed_docs = []
    with open(test_sgtb_path, 'rb') as test_sgtb_file:
        for line in test_sgtb_file:
            test_processed_docs.append(json.loads(line.strip()))

    train_dataset = processed_docs_to_xgboost_dataset(train_processed_docs)
    dev_dataset = processed_docs_to_xgboost_dataset(dev_processed_docs)
    test_dataset = processed_docs_to_xgboost_dataset(test_processed_docs)

    train_dmatrix = xgb.DMatrix(data=train_dataset[0], label=train_dataset[1])
    dev_dmatrix = xgb.DMatrix(data=dev_dataset[0], label=dev_dataset[1])
    test_dmatrix = xgb.DMatrix(data=test_dataset[0], label=test_dataset[1])
    np.set_printoptions(threshold=np.nan)
    def eval_metric(preds, d):
        labels = d.get_label()
        return 'roc', -1*roc_auc_score(labels, preds)

    train_dev = (np.vstack((train_dataset[0], dev_dataset[0])), np.hstack((train_dataset[1], dev_dataset[1])))
    val_fold = np.hstack((-1*np.ones(train_dataset[0].shape[0]), np.zeros(dev_dataset[0].shape[0])))
    predefined_fold = PredefinedSplit(val_fold.astype(int))

    # specify parameters and distributions to sample from
    param_dist = {"max_depth": range(2, 8),
                  "learning_rate": [10**x for x in range(-1, 0)],
                  "n_estimators": range(1, 50),
                  "colsample_by_tree": np.linspace(0.1, 0.5, 5),
                  "min_child_weight": range(5, 11)}

    base_clf = xgb.XGBClassifier(objective="binary:logistic",
                                silent=True)

    seed = 2345345
    n_iter_search = 100
    search_clf = RandomizedSearchCV(base_clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=predefined_fold,
                                       n_jobs=-1, verbose=2, random_state=seed)
    search_clf.fit(train_dev[0], train_dev[1])

    # xgb_clf.fit(train_dataset[0],
    #            train_dataset[1],
    #            eval_set = [(train_dataset[0], train_dataset[1]), (dev_dataset[0], dev_dataset[1])],
    #            eval_metric=eval_metric,
    #            verbose=True,
    #            early_stopping_rounds=10)
    print(search_clf.best_score_)
    print(search_clf.best_params_)

    xgb_clf = xgb.XGBClassifier(objective="binary:logistic",
                                silent=True,
                                **search_clf.best_params_)
    xgb_clf.fit(train_dataset[0], train_dataset[1])
    print(xgb_clf.feature_importances_)

    test_pred = xgb_clf.predict_proba(test_dataset[0])
    dev_pred = xgb_clf.predict_proba(dev_dataset[0])
    train_pred = xgb_clf.predict_proba(train_dataset[0])
    test_pred_threshold = xgb_clf.predict(test_dataset[0])
    dev_pred_threshold = xgb_clf.predict(dev_dataset[0])
    train_pred_threshold = xgb_clf.predict(train_dataset[0])
    test_pred = [probs[1] for probs in test_pred]
    dev_pred = [probs[1] for probs in dev_pred]
    train_pred = [probs[1] for probs in train_pred]

    print(roc_auc_score(test_dataset[1], test_pred))
    print(roc_auc_score(dev_dataset[1], dev_pred))
    print(roc_auc_score(train_dataset[1], train_pred))
    print("test")
    print(precision_score(test_dataset[1], test_pred_threshold))
    print(recall_score(test_dataset[1], test_pred_threshold))
    print(f1_score(test_dataset[1], test_pred_threshold))
    # print(precision_recall_curve(test_dataset[1], test_pred))
    print("dev")
    print(precision_score(dev_dataset[1], dev_pred_threshold))
    print(recall_score(dev_dataset[1], dev_pred_threshold))
    print(f1_score(dev_dataset[1], dev_pred_threshold))
    print("train")
    print(precision_score(train_dataset[1], train_pred_threshold))
    print(recall_score(train_dataset[1], train_pred_threshold))
    print(f1_score(train_dataset[1], train_pred_threshold))
    print(xgb_clf.feature_importances_)

    model_name = "xgboost_linking_model_v24"
    output_model_path = os.path.abspath(os.path.join("project", "linking_models", model_name + ".pkl"))
    joblib.dump({"clf": xgb_clf}, output_model_path)
    output_model_path_full = os.path.abspath(os.path.join("project", "linking_models", model_name + "_full.pkl"))
    xgb_clf.fit(train_dev[0], train_dev[1])
    test_pred_threshold = xgb_clf.predict(test_dataset[0])
    dev_pred_threshold = xgb_clf.predict(dev_dataset[0])
    train_pred_threshold = xgb_clf.predict(train_dataset[0])
    print(f1_score(train_dataset[1], train_pred_threshold))
    print(f1_score(dev_dataset[1], dev_pred_threshold))
    print(f1_score(test_dataset[1], test_pred_threshold))
    joblib.dump({"clf": xgb_clf}, output_model_path_full)

    train_dev_test = (np.vstack((train_dataset[0], dev_dataset[0], test_dataset[0])), np.hstack((train_dataset[1], dev_dataset[1], test_dataset[1])))
    output_model_path_full_test = os.path.abspath(os.path.join("project", "linking_models", model_name + "_full_test.pkl"))
    xgb_clf.fit(train_dev_test[0], train_dev_test[1])
    test_pred_threshold = xgb_clf.predict(test_dataset[0])
    dev_pred_threshold = xgb_clf.predict(dev_dataset[0])
    train_pred_threshold = xgb_clf.predict(train_dataset[0])
    print(f1_score(train_dataset[1], train_pred_threshold))
    print(f1_score(dev_dataset[1], dev_pred_threshold))
    print(f1_score(test_dataset[1], test_pred_threshold))
    joblib.dump({"clf": xgb_clf}, output_model_path_full_test)

if __name__ == "__main__":
    main()