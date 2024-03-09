def train(data, model, optimizer, loss_fn):
    model.train()
    tokens_ids, labels, masks = data
    outputs = model(tokens_ids, masks)
    loss = loss_fn(outputs, labels)
    preds = outputs.argmax(-1)
    labels = labels.argmax(-1)
    acc = (sum(preds==labels) / len(labels))
    model.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, acc