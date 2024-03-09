import torch
@torch.no_grad()
def validate(data, model, loss_fn):
    model.eval()
    tokens_ids, labels, masks = data
    outputs = model(tokens_ids, masks)
    loss = loss_fn(outputs, labels)
    preds = outputs.argmax(-1)
    labels = labels.argmax(-1)
    acc = (sum(preds==labels) / len(labels))
    total_predict.extend(list(preds.cpu().numpy()))
    total_label.extend(list(labels.cpu().numpy()))
    return loss, acc