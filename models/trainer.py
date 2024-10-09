import torch
import Utils

class DGCNNTrainer(object):
    pass

def getTrainer(args):
    if args.cpu == True:
        deviceUsing = torch.device('cpu')
    else:
        deviceUsing = torch.device('cuda:%d' % args.deviceIndex)
    edgeIndex, edgeWeight = Utils.getEdgeWeight(args)
    return DGCNNTrainer(
            edgeIndex = edgeIndex,
            edgeWeight = edgeWeight,
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