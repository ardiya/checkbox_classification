from pytorch_metric_learning import miners, losses
from torch import cat
from torch.nn import CrossEntropyLoss

import torch.optim as optim


class Trainer:
    def __init__(self, cfg, net):
        self.net = net
        self.cfg = cfg
        self.miner = miners.TripletMarginMiner()
        self.triplet_loss_fn = losses.TripletMarginLoss()
        self.cls_loss_fn = CrossEntropyLoss()
        self.optimizer = optim.Adam(
            net.parameters(), weight_decay=cfg["train"]["weight_decay"]
        )

    def fit(self, images, labels):
        self.optimizer.zero_grad()
        embeddings = list()
        predictions = list()
        for image in images:
            embedding, prediction = self.net(image.unsqueeze(0))
            embeddings.append(embedding)
            predictions.append(prediction)
        embeddings = cat(embeddings, dim=0)
        predictions = cat(predictions, dim=0)

        hard_pairs = self.miner(embeddings, labels)
        triplet_loss = self.cfg["train"]["weight_triplet"] * self.triplet_loss_fn(
            embeddings, labels, hard_pairs
        )
        classification_loss = self.cfg["train"]["weight_cls"] * self.cls_loss_fn(
            predictions, labels
        )
        loss = triplet_loss + classification_loss
        loss.backward()
        self.optimizer.step()

        return classification_loss, triplet_loss, self.miner.num_triplets
