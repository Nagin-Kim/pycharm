import  parser
import torch
import math
from torch.nn.parameter import Parameter

args=parser.parse_args()
args.cuda=not args.no_cuda and torch.cuda.is_avaaliable()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#加载数据
adj,feature,labels,idx_train,idx_val,idx_test=load_data()

model=GCN(nfeat=feature.shape[1],
          mid=args.hidden,
          nclass=labels.max().item()+1,
          dropout=args.dropout)
optimizer=optimizer.Adamson(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
class GCN(torch.nn.):

def train(epoch):
