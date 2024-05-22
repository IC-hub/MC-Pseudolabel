import logging
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.special import softmax
from register import Register


register_obj = Register('metric_register')

def compute_metrics(predict_matrix, target_matrix, metrics):
    metric_dict = {}
    for metric in metrics:
        metric_class = register_obj[metric]
        if metric_class is None:
            logging.warning(f'metric:{metric} is not registered.')
        else:
            metric_value = metric_class.compute(predict_matrix, target_matrix)
            metric_dict[metric] = metric_value
    return metric_dict

@register_obj.register
class RMSE:
    def compute(predict_matrix, target_matrix):
        return np.sqrt( ((predict_matrix - target_matrix) * (predict_matrix - target_matrix)).mean() )

@register_obj.register
class Accuracy:
    def compute(predict_matrix, target_matrix):
        predict_result = np.zeros(len(target_matrix))
        predict_result[predict_matrix > 0] = 1
        return accuracy_score(predict_result, target_matrix)
    
@register_obj.register
class AccuracyWithProb:
    def compute(predict_matrix, target_matrix):
        predict_result = np.zeros(len(target_matrix))
        predict_result[predict_matrix > 0.5] = 1
        return accuracy_score(predict_result, target_matrix)

@register_obj.register
class AccuracyPos:
    def compute(predict_matrix, target_matrix):
        predict_result = np.zeros(len(target_matrix))
        predict_result[predict_matrix > 0] = 1
        pos_postion = (target_matrix == 1)
        predict_result = predict_result[pos_postion]
        target_matrix = target_matrix[pos_postion]
        return accuracy_score(predict_result, target_matrix)

@register_obj.register
class AccuracyNeg:
    def compute(predict_matrix, target_matrix):
        predict_result = np.zeros(len(target_matrix))
        predict_result[predict_matrix > 0] = 1
        neg_postion = (target_matrix == 0)
        predict_result = predict_result[neg_postion]
        target_matrix = target_matrix[neg_postion]
        return accuracy_score(predict_result, target_matrix)

@register_obj.register
class AUC:
    def compute(predict_matrix, target_matrix):
        return roc_auc_score(target_matrix, predict_matrix)

@register_obj.register
class F1:
    def compute(predict_matrix, target_matrix):
        predict_result = np.zeros(len(target_matrix))
        predict_result[predict_matrix > 0] = 1
        return f1_score(target_matrix, predict_result)

@register_obj.register
class F1_macro:
    def compute(predict_matrix, target_matrix):
        predict_result = np.zeros(len(target_matrix))
        predict_result[predict_matrix > 0] = 1
        return f1_score(target_matrix, predict_result, average='macro')

@register_obj.register
class AccuracyMulticlass:
    def compute(predict_matrix, target_matrix):
        predict_result = predict_matrix.argmax(-1)
        return accuracy_score(predict_result, target_matrix)


@register_obj.register
class F1Multiclass_macro:
    def compute(predict_matrix, target_matrix):
        predict_result = predict_matrix.argmax(-1)
        return f1_score(target_matrix, predict_result, average='macro')

@register_obj.register
class AUCMulticlass:
    def compute(predict_matrix, target_matrix):
        probability = softmax(predict_matrix, axis=1)
        print (target_matrix.shape, probability.shape)
        return roc_auc_score(target_matrix, probability, average='macro', multi_class='ovo')


class AccuracyMulticlassForLabel:
    def compute(predict_matrix, target_matrix, label):
        predict_result = predict_matrix.argmax(-1)
        postion = (target_matrix == label)
        predict_result = predict_result[postion]
        target_matrix = target_matrix[postion]
        return accuracy_score(predict_result, target_matrix)


@register_obj.register
class AccuracyMulticlassForLabel0:
    def compute(predict_matrix, target_matrix):
        return AccuracyMulticlassForLabel.compute(predict_matrix, target_matrix, 0)


@register_obj.register
class AccuracyMulticlassForLabel1:
    def compute(predict_matrix, target_matrix):
        return AccuracyMulticlassForLabel.compute(predict_matrix, target_matrix, 1)

@register_obj.register
class AccuracyMulticlassForLabel2:
    def compute(predict_matrix, target_matrix):
        return AccuracyMulticlassForLabel.compute(predict_matrix, target_matrix, 2)

@register_obj.register
class AccuracyMulticlassForLabel3:
    def compute(predict_matrix, target_matrix):
        return AccuracyMulticlassForLabel.compute(predict_matrix, target_matrix, 3)

@register_obj.register
class AccuracyMulticlassForLabel4:
    def compute(predict_matrix, target_matrix):
        return AccuracyMulticlassForLabel.compute(predict_matrix, target_matrix, 4)

@register_obj.register
class Pearson:
    def compute(predict_matrix, target_matrix):
        return np.corrcoef(predict_matrix, target_matrix)[0, 1]


