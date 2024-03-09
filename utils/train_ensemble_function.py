def train_ensemble(data_1,data_2, model, optimizer, loss_fn):
    model.train()
    tokens_ids_1,labels,masks_1 = data_1
    tokens_ids_2,labels,masks_2 = data_2
    outputs = model(tokens_ids_1, masks_1,tokens_ids_2,masks_2)
    loss = loss_fn(outputs, labels)
    preds = outputs.argmax(-1)
    labels = labels.argmax(-1)
    acc = (sum(preds==labels) / len(labels))
    model.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, acc