def calculate_accuracy(pred, gt):
    correct = (pred == gt).sum()
    total = len(gt)

    accuracy = correct/total

    return accuracy



