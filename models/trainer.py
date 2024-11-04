import torch
import Utils

class DGCNNTrainer(object):
    pass

def getTrainer(args):
    if args.cpu == True:
        deviceUsing = torch.device('cpu')
    else:
        deviceUsing = torch.device('cuda:%d' % args.deviceIndex)
    edgeIndex, edge_weight = Utils.getedge_weight(args)
    return DGCNNTrainer(
            edgeIndex = edgeIndex,
            edge_weight = edge_weight,
            numClasses=args.numClasses,
            device=deviceUsing,
            numHiddens = args.numHiddens,
            numLayers = args.numLayers,
            dropout = args.dropout,
            barchSize = args.batchSize,
            lr = args.lr,
            l2Reg = args.l2Reg,
            numEpochs = args.numEpochs
        )