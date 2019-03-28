#coding=utf-8
import argparse
import fileinput
import numpy as np
import traceback

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path, skip_header = 0):
    """
    Load data
    """
    row = 0
    features = []
    labels = []
    for line in fileinput.input(file_path):
        if row < skip_header:
            row += 1
            continue
        row += 1
        line = line.strip()
        try:
            shopname, poiname, distance_score, cluster_num, rank_num, name_distance, remove_name_distance, cat_rec, reliability_level, label = line.split("\t")
        except:
            traceback.print_exc()
            continue
        if "\tnan\t" in line:
            continue
        distance = float(distance_score)
        cluster = float(cluster_num)
        rank = float(rank_num)
        name = float(name_distance)
        remove = float(remove_name_distance)
        # shopcat_list = map(int, shopcat.split(','))
        cat_list = map(int, map(float, cat_rec.split(',')))
        reliability = map(int, map(float, reliability_level.split(',')))
        l = int(label)
        sub_feature = []
        sub_feature.append(distance)
        sub_feature.append(cluster)
        sub_feature.append(rank)
        sub_feature.append(name)
        sub_feature.append(remove)
        sub_feature.extend(cat_list)
        sub_feature.extend(reliability)
        features.append(sub_feature)
        labels.append(l)
    return features, labels

def split_data(feature, k_fold = 10):
    """
    Split data to k_fold
    """
    kf = KFold(n_splits = k_fold)
    return kf.split(feature)

def get_eva_results(y_pre, y_test):
    if len(y_pre) != len(y_test):
        return 0.0, 0.0, 0.0
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for i, value in enumerate(y_pre):
        if y_test[i] == 1:
            if y_pre[i] == 1:
                tp += 1
            else:
                fn += 1
        else: # y_test = 0
            if y_pre[i] == 1:
                fp += 1
            else:
                tn += 1
    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def write_output(x_test, y_test, y_pre, test, epoch, out_dir):
    fout = open(out_dir + str(epoch) + ".out", "wb")
    x_array = np.array(x_test)
    rows = x_array.shape[0]
    ytest_array = np.array(y_test).reshape(rows, 1)
    y_pre_array = np.array(y_pre).reshape(rows, 1)
    test_array = np.array(test).reshape(rows, 1)
    out = np.concatenate((test_array, x_array, ytest_array, y_pre_array), axis = 1).tolist()

    fout.write("\n".join("\t".join(str(x) for x in sub_out) for sub_out in out))
    fout.close()

def write_test_output(x_test, y_test, y_pre, epoch, out_dir):
    fout = open(out_dir + str(epoch) + ".test", "wb")
    x_array = np.array(x_test)
    rows = x_array.shape[0]
    ytest_array = np.array(y_test).reshape(rows, 1)
    y_pre_array = np.array(y_pre).reshape(rows, 1)
    out = np.concatenate((x_array, ytest_array, y_pre_array), axis = 1).tolist()
    fout.write("\n".join("\t".join(str(x) for x in sub_out) for sub_out in out))
    fout.close()

def evaluate(gb_model, lr_model, onehot_model, x_test, y_test, epoch, test, out_dir):
    y_pre = lr_model.predict(onehot_model.transform(gb_model.apply(x_test)[:, :, 0]))
    precision, recall, f1 = get_eva_results(y_pre, y_test)
    results["precision"][epoch - 1] = precision
    results["recall"][epoch - 1] = recall
    results["f1"][epoch - 1] = f1
    write_output(x_test, y_test, y_pre, test, epoch, out_dir)
    print ("Epoch %d, precision %f" % (epoch, precision))
    print ("Epoch %d, recall %f" % (epoch, recall))
    print ("Epoch %d, f1 %f" % (epoch, f1))

def test_model(gb_model, lr_model, onehot_model, test_x, test_y, epoch, out_dir):
    y_pre = lr_model.predict(onehot_model.transform(gb_model.apply(test_x)[:, :, 0]))
    precision, recall, f1 = get_eva_results(y_pre, test_y)
    test_results["precision"][epoch - 1] = precision
    test_results["recall"][epoch - 1] = recall
    test_results["f1"][epoch - 1] = f1
    write_test_output(test_x, test_y, y_pre, epoch, out_dir)
    print ("Testing epoch %d, precision %f" % (epoch, precision))
    print ("Testing epoch %d, recall %f" % (epoch, recall))
    print ("Testing epoch %d, f1 %f" % (epoch, f1))

def splice_array(x, indices):
    rets = []
    for num in indices:
        rets.append(x[num])
    return rets

def train(file_path, test_path, out_dir, cross_split = 10, estimators = 100, min_samples_split = 5, learning_rate = 0.05, max_iter = 1000):
    """
    Train step
    """
    print "Loading data..."
    features, labels = load_data(file_path)
    test_features, test_labels = load_data(test_path)

    print "Split training data..."
    split = split_data(features, cross_split)
    epoch = 1

    results = {}
    global results
    results["precision"] = np.zeros(cross_split)
    results["recall"] = np.zeros(cross_split)
    results["f1"] = np.zeros(cross_split)

    test_results = {}
    global test_results
    test_results["precision"] = np.zeros(cross_split)
    test_results["recall"] = np.zeros(cross_split)
    test_results["f1"] = np.zeros(cross_split)

    for train, test in split: # cross validation
        print ("Cross validation term %d ..." % epoch)
        print ("Training data split %d/%d ..." % (len(train), len(test)))
        feature_train = splice_array(features, train)
        feature_test = splice_array(features, test)
        label_train = splice_array(labels, train)
        label_test = splice_array(labels, test)
        feature_train, feature_train_lr, label_train, label_train_lr = train_test_split(feature_train, label_train, test_size = 0.2)


        """
        Train model
        """
        print ("Fitting model...")
        params = {
           "n_estimators": estimators,
           "min_samples_split": min_samples_split,
           "learning_rate": learning_rate
        }

        gb = GradientBoostingClassifier(**params)

        gb_enc = OneHotEncoder()

        params = {
            "solver" : "lbfgs",
            "max_iter": max_iter
        }
        gb_lr = LogisticRegression(**params)

        gb.fit(feature_train, label_train)
        gb_enc.fit(gb.apply(feature_train)[:, :, 0])
        gb_lr.fit(gb_enc.transform(gb.apply(feature_train_lr)[:, :, 0]), label_train_lr)

        # Evaluate model
        print ("Evaluating model ...")
        evaluate(gb, gb_lr, gb_enc, feature_test, label_test, epoch, test, out_dir)

        print ("Testing model ...")
        test_model(gb, gb_lr, gb_enc, test_features, test_labels, epoch, out_dir)

        epoch += 1

    print ("Overall precision %f" % (np.mean(results["precision"])))
    print ("Overall recall %f" % (np.mean(results["recall"])))
    print ("Overall f1 %f" % (np.mean(results["f1"])))

    print ("Overall test precision %f" % (np.mean(test_results["precision"])))
    print ("Overall test recall %f" % (np.mean(test_results["recall"])))
    print ("Overall test f1 %f" % (np.mean(test_results["f1"])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path")
    parser.add_argument("--test_path")
    parser.add_argument("--outdir_path")
    parser.add_argument("--estimators", default = 100, type = int)
    parser.add_argument("--min_samples_split", default = 5, type = int)
    parser.add_argument("--learning_rate", default = 0.05, type = float)
    parser.add_argument("--cross_split", default = 10, type = int)
    parser.add_argument("--max_iter", default = 1000, type = int)
    args = parser.parse_args()
    file_path = args.file_path
    test_path = args.test_path

    if not os.path.exists("out"):
        os.mkdir("out")

    train(file_path, test_path, "out/", args.cross_split, args.estimators, args.min_samples_split, args.learning_rate, args.max_iter)
