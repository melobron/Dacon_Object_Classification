from sklearn.metrics import accuracy_score, recall_score,



def cal_acc_recal_pre_f1(outputs, targets):
    target = targets.reshape(-1)
    output = outputs.reshape(-1)
    acc = accuracy_score(target, output)
    recall = recall_score(target, output, average='macro', zero_division = 0)
    precision = precision_score(target, output, average='macro', zero_division = 0)
    F1_score = f1_score(target, output, average='macro', zero_division = 0)
    return acc, recall, precision, F1_score

