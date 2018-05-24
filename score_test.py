import numpy as np

def mean_iou_score(y_pred, y_true):
    class_iou_list = []
    labels = set(y_true).union(y_pred)
    for i in labels:
        intersect = np.sum(np.logical_and((y_pred == i), (y_true == i)))
        union = np.sum(np.logical_or((y_pred == i), (y_true == i)))
        class_iou_list.append(intersect * 1. / union)

    return np.mean(class_iou_list)

def mean_iou(label, prediction):
    iou_list = []
    for i in range(len(label)):
        score = mean_iou_score(label[i], prediction[i])
        iou_list.append(score)

    return np.mean(iou_list)

def accuracy_score(y_true, y_pred):
    p = np.sum(y_true == y_pred)
    return p * 1.0 / len(y_true)

def accuracy(label, prediction):
    accuracy_list = []
    for i in range(len(label)):
        score = accuracy_score(label[i], prediction[i])
        accuracy_list.append(score)

    return np.mean(accuracy_list)

def f1_score(y_true, y_pred):
    tp = np.sum(np.logical_and(y_true, y_pred))
    #tn = np.sum(np.logical_and(y_true == False, y_pred=False))
    fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
    fn = np.sum(np.logical_and(y_true, np.logical_not(y_pred)))

    if tp + fp != 0:
        p = (1. * tp) / (tp + fp)
    else:
        p = 1

    if tp + fn != 0:
        r = (1. * tp) / (tp + fn)
    else:
        r =0

    if p + r > 0:
        f = 2 * p * r / (p + r)
    else:
        f = 0

    return f

def fscore(label, prediction):
    fscore_list = []
    for i in range(len(label)):
        score = f1_score(label[i], prediction[i])
        fscore_list.append(score)

    return np.mean(fscore_list)

def run_test():
    prediction_batch = np.array([[0,1,0,1], [1,0,1,0],[0,1,0,0],[0,0,0,1]])
    true_batch = np.array([[1,0,1,0],[0,1,0,1],[0,1,0,0],[0,0,0,1]])

    class_pred_batch = np.array([[0,0,0,0, 1,0,0,1, 1,2,2,1, 3,3,0,3]])
    class_true_batch = np.array([[0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]])

    f = fscore(true_batch == 1, prediction_batch==1)
    a = accuracy(true_batch, prediction_batch)
    i = mean_iou(class_true_batch, class_pred_batch)

    print("F-score: %.3f" % f)
    print("accuracy: %.3f" % a)
    print("mean-iou: %.3f" % i)

if __name__ == '__main__':
    run_test()
