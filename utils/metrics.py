import torch

def accuracy(pred,gt):
    """
    Compute the accuracy for a binary classification problem
    Arguments:
        pred (torch.tensor): prediction of the model
        gt (torch.tensor): ground truth
    Return:
        (float): accuracy
    """
    return torch.sum(pred==gt)/torch.numel(gt)
    
def IoU(pred,gt,label=1):
    """
    Compute the IoU (Intersection over Union) for a specified label in the prediction
    Arguments:
        pred (torch.tensor): prediction of the model
        gt (torch.tensor): ground truth
        label (int): label used in prediction for which we want to compute the IoU
    Return:
        (float): IoU
    """
    pred_mask=pred==label
    gt_mask=gt==label
    union=torch.sum(torch.logical_or(pred_mask,gt_mask))
    intersection=torch.sum(pred_mask*gt_mask)
    if union==0:
        return None
    return intersection/union

def precision(pred,gt,label=1):
    """
    Compute the precision for a specified label in the prediction
    Arguments:
        pred (torch.tensor): prediction of the model
        gt (torch.tensor): ground truth
        label (int): label used in prediction for which we want to compute the IoU
    Return:
        (float): precision
    """
    pred_mask=pred==label
    gt_mask=gt==label
    TP=torch.sum(pred_mask*gt_mask)
    FP=torch.sum(pred_mask)-TP
    if TP+FP==0:
        return None
    return TP/(TP+FP)

def recall(pred,gt,label=1):
    """
    Compute the recall for a specified label in the prediction
    Arguments:
        pred (torch.tensor): prediction of the model
        gt (torch.tensor): ground truth
        label (int): label used in prediction for which we want to compute the IoU
    Return:
        (float): recall
    """
    pred_mask=pred==label
    gt_mask=gt==label
    TP=torch.sum(pred_mask*gt_mask)
    FN=torch.sum(gt_mask)-TP
    if TP+FN==0:
        return None
    return TP/(TP+FN)

def f1_score(pred,gt,label=1):
    """
    Compute the f1_score for a specified label in the prediction
    Arguments:
        pred (torch.tensor): prediction of the model
        gt (torch.tensor): ground truth
        label (int): label used in prediction for which we want to compute the IoU
    Return:
        (float): f1_score
    """
    prec=precision(pred,gt,label)
    rec=recall(pred,gt,label)
    if not (prec and rec):
        return None
    return (2*prec*rec)/(prec+rec)