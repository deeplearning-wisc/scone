# -*- coding: utf-8 -*-
from sklearn.metrics import det_curve, accuracy_score, roc_auc_score
from make_datasets import *
from models.wrn_ssnd import *

from models.mlp import *

import wandb

import seaborn as sns
import matplotlib.pyplot as plt

# for t-sne plot
from time import time
import pandas as pd

from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import plotly.graph_objects as go


from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import sklearn.metrics as sk

#import warning
#warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

'''
This code implements training and testing functions. 
'''


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'MNIST'],
                    default='MNIST', # default cifar10
                    help='Choose between CIFAR-10, CIFAR-100, MNIST.')
parser.add_argument('--model', '-m', type=str, default='mlp', # default allconv
                    choices=['allconv', 'wrn', 'densenet', 'mlp'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float,
                    default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int,
                    default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int,
                    default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float,
                    default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float,
                    help='dropout probability')
# Checkpoints
parser.add_argument('--results_dir', type=str,
                    default='results', help='Folder to save .pkl results.')
parser.add_argument('--checkpoints_dir', type=str,
                    default='checkpoints', help='Folder to save .pt checkpoints.')

parser.add_argument('--load_pretrained', type=str, default=None, help='Load pretrained model to test or resume training.')
#parser.add_argument('--load_pretrained', type=str,
#                    default='snapshots/pretrained', help='Load pretrained model to test or resume training.')
parser.add_argument('--test', '-t', action='store_true',
                    help='Test only flag.')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=1, help='Which GPU to run on.')
parser.add_argument('--prefetch', type=int, default=4,
                    help='Pre-fetching threads.')
# EG specific
parser.add_argument('--score', type=str, default='SSND', help='SSND|OE|energy|VOS')
parser.add_argument('--seed', type=int, default=1,
                    help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
parser.add_argument('--classification', type=boolean_string, default=True)

# dataset related
parser.add_argument('--aux_out_dataset', type=str, default='FashionMNIST', choices=['svhn', 'lsun_c', 'lsun_r',
                    'isun', 'dtd', 'places', 'tinyimages_300k', 'FashionMNIST'],
                    help='Auxiliary out of distribution dataset')
parser.add_argument('--test_out_dataset', type=str, default='FashionMNIST', choices=['svhn', 'lsun_c', 'lsun_r',
                    'isun', 'dtd', 'places', 'tinyimages_300k', 'FashionMNIST'],
                    help='Test out of distribution dataset')
parser.add_argument('--pi_1', type=float, default=0.5,
                    help='pi in ssnd framework, proportion of ood data in auxiliary dataset')
parser.add_argument('--pi_2', type=float, default=0.5,
                    help='pi in ssnd framework, proportion of ood data in auxiliary dataset')

parser.add_argument('--pseudo', type=float, default=0.01,
                    help='pseudo regulazier')

parser.add_argument('--start_epoch', type=int, default=50,
                    help='start epoches')

parser.add_argument('--cortype', type=str, default='gaussian_noise', help='corrupted type of images')
###scone/woods/woods_nn specific
parser.add_argument('--in_constraint_weight', type=float, default=1,
                    help='weight for in-distribution penalty in loss function')
parser.add_argument('--out_constraint_weight', type=float, default=1,
                    help='weight for out-of-distribution penalty in loss function')
parser.add_argument('--ce_constraint_weight', type=float, default=1,
                    help='weight for classification penalty in loss function')
parser.add_argument('--false_alarm_cutoff', type=float,
                    default=0.05, help='false alarm cutoff')

parser.add_argument('--lr_lam', type=float, default=1, help='learning rate for the updating lam (SSND_alm)')
parser.add_argument('--ce_tol', type=float,
                    default=2, help='tolerance for the loss constraint')

parser.add_argument('--penalty_mult', type=float,
                    default=1.5, help='multiplicative factor for penalty method')

parser.add_argument('--constraint_tol', type=float,
                    default=0, help='tolerance for considering constraint violated')


parser.add_argument('--eta', type=float, default=1.0, help='woods with margin loss')

parser.add_argument('--alpha', type=float, default=0.05, help='number of labeled samples')

# Energy Method Specific
parser.add_argument('--m_in', type=float, default=-25.,
                    help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-5.,
                    help='margin for out-distribution; below this value will be penalized')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')  # T = 1 suggested by energy paper

#energy vos method
parser.add_argument('--energy_vos_lambda', type=float, default=2, help='energy vos weight')

# OE specific
parser.add_argument('--oe_lambda', type=float, default=.5, help='OE weight')


# parse argument
args = parser.parse_args()

# method_data_name gives path to the model
if args.score in ['woods_nn']:
    method_data_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.score,
                                               str(args.in_constraint_weight),
                                               str(args.out_constraint_weight),
                                               str(args.ce_constraint_weight),
                                               str(args.false_alarm_cutoff),
                                               str(args.lr_lam),
                                               str(args.penalty_mult),
                                               str(args.pi_1),
                                               str(args.pi_2))
elif args.score == "energy":
    method_data_name = "{}_{}_{}_{}_{}".format(args.score,
                                      str(args.m_in),
                                      str(args.m_out),
                                      str(args.pi_1),
                                      str(args.pi_2))
elif args.score == "OE":
    method_data_name = "{}_{}_{}_{}".format(args.score,
                                   str(args.oe_lambda),
                                   str(args.pi_1),
                                   str(args.pi_2))
elif args.score == "energy_vos":
    method_data_name = "{}_{}_{}_{}".format(args.score,
                                   str(args.energy_vos_lambda),
                                   str(args.pi_1),
                                   str(args.pi_2))
elif args.score in ['scone', 'woods']:
    method_data_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.score,
                                            str(args.in_constraint_weight),
                                            str(args.out_constraint_weight),
                                            str(args.false_alarm_cutoff),
                                            str(args.ce_constraint_weight),
                                            str(args.lr_lam),
                                            str(args.penalty_mult),
                                            str(args.pi_1),
                                            str(args.pi_2))


state = {k: v for k, v in args._get_kwargs()}
print(state)

#save wandb hyperparameters
# wandb.config = state

wandb.init(project="oodproject", entity="ood_learning", config=state)
state['wandb_name'] = wandb.run.name

# store train, test, and valid FPR95
state['fpr95_train'] = []
state['fpr95_valid'] = []
state['fpr95_valid_clean'] = []
state['fpr95_test'] = []

state['auroc_train'] = []
state['auroc_valid'] = []
state['auroc_valid_clean'] = []
state['auroc_test'] = []

state['val_wild_total'] = []
state['val_wild_class_as_in'] = []

# in-distribution classification accuracy
state['train_accuracy'] = []
state['valid_accuracy'] = []
state['valid_accuracy_clean'] = []
state['valid_accuracy_cor'] = []
state['valid_accuracy_clean_cor'] = []
state['test_accuracy'] = []
state['test_accuracy_cor'] = []

# store train, valid, and test OOD scores
state['OOD_scores_P0_train'] = []
state['OOD_scores_PX_train'] = []
state['OOD_scores_P0_valid'] = []
state['OOD_scores_PX_valid'] = []
state['OOD_scores_P0_valid_clean'] = []
state['OOD_scores_PX_valid_clean'] = []
state['OOD_scores_P0_test'] = []
state['OOD_scores_Ptest'] = []

# optimization constraints
state['in_dist_constraint'] = []
state['train_loss_constraint'] = []

def to_np(x): return x.data.cpu().numpy()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

#make the data_loaders
train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out, test_loader_in, test_loader_cor, test_loader_ood, valid_loader_in, valid_loader_aux = make_datasets(args.dataset, args.aux_out_dataset, args.test_out_dataset, state, args.alpha, args.pi_1, args.pi_2, args.cortype)


print("\n len(train_loader_in.dataset) {} " \
      "len(train_loader_aux_in.dataset) {}, " \
      "len(train_loader_aux_in_cor.dataset) {}, "\
      "len(train_loader_aux_out.dataset) {}, " \
      "len(test_loader_mnist.dataset) {}, " \
      "len(test_loader_cor.dataset) {}, " \
      "len(test_loader_ood.dataset) {}, " \
      "len(valid_loader_in.dataset) {}, " \
      "len(valid_loader_aux.dataset) {}".format(
    len(train_loader_in.dataset),
    len(train_loader_aux_in.dataset),
    len(train_loader_aux_in_cor.dataset),
    len(train_loader_aux_out.dataset),
    len(test_loader_in.dataset),
    len(test_loader_cor.dataset),
    len(test_loader_ood.dataset),
    len(valid_loader_in.dataset),
    len(valid_loader_aux.dataset)))

state['train_in_size'] = len(train_loader_in.dataset)
state['train_aux_in_size'] = len(train_loader_aux_in.dataset)
state['train_aux_out_size'] = len(train_loader_aux_out.dataset)
state['valid_in_size'] = len(valid_loader_in.dataset)
state['valid_aux_size'] = len(valid_loader_aux.dataset)
state['test_in_size'] = len(test_loader_in.dataset)
state['test_in_cor_size'] = len(test_loader_cor.dataset)
state['test_out_size'] = len(test_loader_ood.dataset)

if args.dataset in ['cifar10']:
    num_classes = 10
elif args.dataset in ['cifar100']:
    num_classes = 100
elif args.dataset in ['MNIST']:
    num_classes = 10  

# WRN architecture with 10 output classes (extra NN is added later for SSND methods)
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
# MLP architecture with num_classes
#net = woods_mlp(num_classes)


# create logistic regression layer for energy_vos and woods
if args.score in ['energy_vos', 'woods', 'scone']:
    logistic_regression = nn.Linear(1, 1)
    logistic_regression.cuda()


# Restore model
model_found = False
print(args.load_pretrained)
if args.load_pretrained == 'snapshots/pretrained':
    print('Restoring trained model...')
    for i in range(200, -1, -1):

        model_name = os.path.join(args.load_pretrained, args.dataset + '_' + args.model +
                                  '_pretrained_epoch_' + str(i) + '.pt')
        model_name = os.path.join(args.load_pretrained, 'in_ratio_50_cifar100_wrn_pretrained_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            print('found pretrained model: {}'.format(model_name))
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            model_found = True
            break
    if not model_found:
        assert False, "could not find model to restore"



# add extra NN for OOD detection (for SSND methods)
if args.score in ['woods_nn']:
    net = WideResNet_SSND(wrn=net)
    #net = woods_mlp(num_classes)

if args.ngpu > 1:
    print('Available CUDA devices:', torch.cuda.device_count())
    print('CUDA available:', torch.cuda.is_available())
    print('Running in parallel across', args.ngpu, 'GPUs')
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    net.cuda()
    torch.cuda.manual_seed(1)
elif args.ngpu > 0:
    print('CUDA available:', torch.cuda.is_available())
    print('Available CUDA devices:', torch.cuda.device_count())
    print('Sending model to device', torch.cuda.current_device(), ':', torch.cuda.get_device_name())
    net.cuda()
    torch.cuda.manual_seed(1)

# cudnn.benchmark = True  # fire on all cylinders
cudnn.benchmark = False  # control reproducibility/stochastic behavior

#energy_vos, woods also use logistic regression in optimization
if args.score in ['energy_vos', 'woods', 'scone']:
    optimizer = torch.optim.SGD(
        list(net.parameters()) + list(logistic_regression.parameters()),
        state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)

else:
    optimizer = torch.optim.SGD(
        net.parameters(), state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)

#define scheduler for learning rate
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[int(args.epochs*.5), int(args.epochs*.75), int(args.epochs*.9)], gamma=0.5)

# /////////////// Training ///////////////

# Create extra variable needed for training

# make in_constraint a global variable
in_constraint_weight = args.in_constraint_weight

# make loss_ce_constraint a global variable
ce_constraint_weight = args.ce_constraint_weight

# create the lagrangian variable for lagrangian methods
if args.score in ['woods_nn', 'woods', 'scone']:
    lam = torch.tensor(0).float()
    lam = lam.cuda()

    lam2 = torch.tensor(0).float()
    lam2 = lam.cuda()


def mix_batches(aux_in_set, aux_in_cor_set, aux_out_set):
    '''
    Args:
        aux_in_set: minibatch from in_distribution
        aux_in_cor_set: minibatch from covariate shift OOD distribution
        aux_out_set: minibatch from semantic shift OOD distribution

    Returns:
        mixture of minibatches with mixture proportion pi_1 of aux_in_cor_set and pi_2 of aux_out_set
    '''
    
    # create a mask to decide which sample is in the batch
    mask_1 = rng.choice(a=[False, True], size=(args.batch_size,), p=[1 - args.pi_1, args.pi_1])
    aux_in_cor_set_subsampled = aux_in_cor_set[0][mask_1]

    mask_2 = rng.choice(a=[False, True], size=(args.batch_size,), p=[1 - args.pi_2, args.pi_2])
    aux_out_set_subsampled = aux_out_set[0][mask_2]

    mask_12 = rng.choice(a=[False, True], size=(args.batch_size,), p=[1 - (args.pi_1 + args.pi_2), (args.pi_1 + args.pi_2)])
    #mask = rng.choice(a=[False, True], size=(args.batch_size,), p=[1 - 0.05, 0.05])
    aux_in_set_subsampled = aux_in_set[0][np.invert(mask_12)]

    # note: ordering of aux_out_set_subsampled, aux_in_set_subsampled does not matter because you always take the sum
    aux_set = torch.cat((aux_out_set_subsampled, aux_in_set_subsampled, aux_in_cor_set_subsampled), 0)

    return aux_set

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])



def train(epoch):
    '''
    Train the model using the specified score
    '''

    # make the variables global for optimization purposes
    global in_constraint_weight
    global ce_constraint_weight

    # declare lam global
    if args.score in ['woods_nn',  'woods', 'scone']:
        global lam
        global lam2

    # print the learning rate
    for param_group in optimizer.param_groups:
        print("lr {}".format(param_group['lr']))

    net.train()  # enter train mode

    # track train classification accuracy
    train_accuracies = []

    # # start at a random point of the dataset for; this induces more randomness without obliterating locality
    train_loader_aux_in.dataset.offset = rng.integers(
        len(train_loader_aux_in.dataset))

    train_loader_aux_in_cor.dataset.offset = rng.integers(
        len(train_loader_aux_in_cor.dataset))   

    train_loader_aux_out.dataset.offset = rng.integers(
        len(train_loader_aux_out.dataset))
    batch_num = 1
    #loaders = zip(train_loader_in, train_loader_aux_in, train_loader_aux_out)
    loaders = zip(train_loader_in, train_loader_aux_in, train_loader_aux_in_cor, train_loader_aux_out)

    # for logging in weights & biases
    losses_ce = []
    in_losses = []
    out_losses = []
    out_losses_weighted = []
    losses = []

    for in_set, aux_in_set, aux_in_cor_set, aux_out_set in loaders:    
        #create the mixed batch
        aux_set = mix_batches(aux_in_set, aux_in_cor_set, aux_out_set)

        batch_num += 1
        data = torch.cat((in_set[0], aux_set), 0)
        target = in_set[1]

        if args.ngpu > 0:
            data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # in-distribution classification accuracy
        if args.score in ['woods_nn']:
            x_classification = x[:len(in_set[0]), :num_classes]
        elif args.score in ['energy', 'OE', 'energy_vos', 'woods', 'scone']:
            x_classification = x[:len(in_set[0])]
        pred = x_classification.data.max(1)[1]
        train_accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

        optimizer.zero_grad()

        # cross-entropy loss
        if args.classification:
            loss_ce = F.cross_entropy(x_classification, target)
        else:
            loss_ce = torch.Tensor([0]).cuda()

        losses_ce.append(loss_ce.item())

        if args.score == 'woods_nn':
            '''
            This is the same as woods_nn but it now uses separate
            weight for in distribution scores and classification scores.

            it also updates the weights separately
            '''

            # penalty for the mixture/auxiliary dataset
            out_x_ood_task = x[len(in_set[0]):, num_classes]
            out_loss = torch.mean(F.relu(1 - out_x_ood_task))
            out_loss_weighted = args.out_constraint_weight * out_loss

            in_x_ood_task = x[:len(in_set[0]), num_classes]
            f_term = torch.mean(F.relu(1 + in_x_ood_task)) - args.false_alarm_cutoff
            if in_constraint_weight * f_term + lam >= 0:
                in_loss = f_term * lam + in_constraint_weight / 2 * torch.pow(f_term, 2)
            else:
                in_loss = - torch.pow(lam, 2) * 0.5 / in_constraint_weight

            loss_ce_constraint = loss_ce - args.ce_tol * full_train_loss
            if ce_constraint_weight * loss_ce_constraint + lam2 >= 0:
                loss_ce = loss_ce_constraint * lam2 + ce_constraint_weight / 2 * torch.pow(loss_ce_constraint, 2)
            else:
                loss_ce = - torch.pow(lam2, 2) * 0.5 / ce_constraint_weight

            # add the losses together
            loss = loss_ce + out_loss_weighted + in_loss

            in_losses.append(in_loss.item())
            out_losses.append(out_loss.item())
            out_losses_weighted.append(out_loss.item())
            losses.append(loss.item())

        elif args.score == 'energy':

            Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1)
            Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
            loss_energy = 0.1 * (torch.pow(F.relu(Ec_in - args.m_in), 2).mean() + torch.pow(F.relu(args.m_out - Ec_out),
                                                                                            2).mean())
            loss = loss_ce + loss_energy

            losses.append(loss.item())

        elif args.score == 'energy_vos':

            Ec_out = torch.logsumexp(x[len(in_set[0]):], dim=1)
            Ec_in = torch.logsumexp(x[:len(in_set[0])], dim=1)
            binary_labels = torch.ones(len(x)).cuda()
            binary_labels[len(in_set[0]):] = 0
            loss_energy = F.binary_cross_entropy_with_logits(logistic_regression(
                torch.cat([Ec_in, Ec_out], -1).unsqueeze(1)).squeeze(),
                                                                 binary_labels)

            loss = loss_ce + args.energy_vos_lambda * loss_energy

            losses.append(loss.item())

        elif args.score == 'scone':

            #apply the sigmoid loss
            loss_energy_in =  torch.mean(torch.sigmoid(logistic_regression(
                (torch.logsumexp(x[:len(in_set[0])], dim=1)).unsqueeze(1)).squeeze()))
            loss_energy_out = torch.mean(torch.sigmoid(-logistic_regression(
                (torch.logsumexp(x[len(in_set[0]):], dim=1) - args.eta).unsqueeze(1)).squeeze()))

            #alm function for the in distribution constraint
            in_constraint_term = loss_energy_in - args.false_alarm_cutoff
            if in_constraint_weight * in_constraint_term + lam >= 0:
                in_loss = in_constraint_term * lam + in_constraint_weight / 2 * torch.pow(in_constraint_term, 2)
            else:
                in_loss = - torch.pow(lam, 2) * 0.5 / in_constraint_weight

            #alm function for the cross entropy constraint
            loss_ce_constraint = loss_ce - args.ce_tol * full_train_loss
            if ce_constraint_weight * loss_ce_constraint + lam2 >= 0:
                loss_ce = loss_ce_constraint * lam2 + ce_constraint_weight / 2 * torch.pow(loss_ce_constraint, 2)
            else:
                loss_ce = - torch.pow(lam2, 2) * 0.5 / ce_constraint_weight

            loss = loss_ce + args.out_constraint_weight*loss_energy_out + in_loss

            #wandb
            in_losses.append(in_loss.item())
            out_losses.append(loss_energy_out.item())
            out_losses_weighted.append(args.out_constraint_weight * loss_energy_out.item())
            losses.append(loss.item())

        elif args.score == 'OE':

            loss_oe = args.oe_lambda * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
            loss = loss_ce + loss_oe
            losses.append(loss.item())

        loss.backward()
        optimizer.step()

    loss_ce_avg = np.mean(losses_ce)
    in_loss_avg = np.mean(in_losses)
    out_loss_avg = np.mean(out_losses)
    out_loss_weighted_avg = np.mean(out_losses_weighted)
    loss_avg = np.mean(losses)
    train_acc_avg = np.mean(train_accuracies)

    wandb.log({
        'epoch':epoch,
        "learning rate": optimizer.param_groups[0]['lr'],
        'CE loss':loss_ce_avg,
        'in loss':in_loss_avg,
        'out loss':out_loss_avg,
        'out loss (weighted)':out_loss_weighted_avg,
        'loss':loss_avg,
        'train accuracy':train_acc_avg
    })

    # store train accuracy
    state['train_accuracy'].append(train_acc_avg)

    # updates for alm methods
    if args.score in ["woods_nn"]:
        print("making updates for SSND alm methods...")

        # compute terms for constraints
        in_term, ce_loss = compute_constraint_terms()

        # update lam for in-distribution term
        if args.score in ["woods_nn"]:
            print("updating lam...")

            in_term_constraint = in_term - args.false_alarm_cutoff
            print("in_distribution constraint value {}".format(in_term_constraint))
            state['in_dist_constraint'].append(in_term_constraint.item())

            # wandb
            wandb.log({"in_term_constraint": in_term_constraint.item(),
                       'in_constraint_weight':in_constraint_weight,
                       'epoch':epoch})

            # update lambda
            if in_term_constraint * in_constraint_weight + lam >= 0:
                lam += args.lr_lam * in_term_constraint
            else:
                lam += -args.lr_lam * lam / in_constraint_weight

        # update lam2
        if args.score in ["woods_nn"]:
            print("updating lam2...")

            ce_constraint = ce_loss - args.ce_tol * full_train_loss
            print("cross entropy constraint {}".format(ce_constraint))
            state['train_loss_constraint'].append(ce_constraint.item())

            # wandb
            wandb.log({"ce_term_constraint": ce_constraint.item(),
                       'ce_constraint_weight':ce_constraint_weight,
                       'epoch':epoch})

            # update lambda2
            if ce_constraint * ce_constraint_weight + lam2 >= 0:
                lam2 += args.lr_lam * ce_constraint
            else:
                lam2 += -args.lr_lam * lam2 / ce_constraint_weight

        # update weight for alm_full_2
        if args.score == 'woods_nn' and in_term_constraint > args.constraint_tol:
            print('increasing in_constraint_weight weight....\n')
            in_constraint_weight *= args.penalty_mult

        if args.score == 'woods_nn' and ce_constraint > args.constraint_tol:
            print('increasing ce_constraint_weight weight....\n')
            ce_constraint_weight *= args.penalty_mult

    #alm update for energy_vos alm methods
    if args.score in ['scone', 'woods']:
        print("making updates for energy alm methods...")
        avg_sigmoid_energy_losses, _, avg_ce_loss = evaluate_energy_logistic_loss()

        in_term_constraint = avg_sigmoid_energy_losses -  args.false_alarm_cutoff
        print("in_distribution constraint value {}".format(in_term_constraint))
        state['in_dist_constraint'].append(in_term_constraint.item())

        # update lambda
        print("updating lam...")
        if in_term_constraint * in_constraint_weight + lam >= 0:
            lam += args.lr_lam * in_term_constraint
        else:
            lam += -args.lr_lam * lam / in_constraint_weight

        # wandb
        wandb.log({"in_term_constraint": in_term_constraint.item(),
                   'in_constraint_weight':in_constraint_weight,
                   "avg_sigmoid_energy_losses": avg_sigmoid_energy_losses.item(),
                   'lam': lam,
                   'epoch':epoch})

        # update lam2
        if args.score in ['scone', 'woods']:
            print("updating lam2...")

            ce_constraint = avg_ce_loss - args.ce_tol * full_train_loss
            print("cross entropy constraint {}".format(ce_constraint))
            state['train_loss_constraint'].append(ce_constraint.item())

            # wandb
            wandb.log({"ce_term_constraint": ce_constraint.item(),
                       'ce_constraint_weight':ce_constraint_weight,
                       'epoch':epoch})

            # update lambda2
            if ce_constraint * ce_constraint_weight + lam2 >= 0:
                lam2 += args.lr_lam * ce_constraint
            else:
                lam2 += -args.lr_lam * lam2 / ce_constraint_weight

        # update in-distribution weight for alm
        if args.score in ['scone', 'woods'] and in_term_constraint > args.constraint_tol:
            print("energy in distribution constraint violated, so updating in_constraint_weight...")
            in_constraint_weight *= args.penalty_mult

        # update ce_loss weight for alm
        if args.score in ['scone', 'woods'] and ce_constraint > args.constraint_tol:
            print('increasing ce_constraint_weight weight....\n')
            ce_constraint_weight *= args.penalty_mult


def compute_constraint_terms():
    '''

    Compute the in-distribution term and the cross-entropy loss over the whole training set
    '''

    net.eval()

    # create list for the in-distribution term and the ce_loss
    in_terms = []
    ce_losses = []
    num_batches = 0
    for in_set in train_loader_in:
        num_batches += 1
        data = in_set[0]
        target = in_set[1]

        if args.ngpu > 0:
            data, target = data.cuda(), target.cuda()

        # forward
        net(data)
        z = net(data)

        # compute in-distribution term
        in_x_ood_task = z[:, num_classes]
        in_terms.extend(list(to_np(F.relu(1 + in_x_ood_task))))

        # compute cross entropy term
        z_classification = z[:, :num_classes]
        loss_ce = F.cross_entropy(z_classification, target, reduction='none')
        ce_losses.extend(list(to_np(loss_ce)))

    return np.mean(np.array(in_terms)), np.mean(np.array(ce_losses))


def compute_fnr(out_scores, in_scores, fpr_cutoff=.05):
    '''
    compute fnr at 05
    '''

    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    fpr, fnr, thresholds = det_curve(y_true=y_true, y_score=y_score)

    idx = np.argmin(np.abs(fpr - fpr_cutoff))

    fpr_at_fpr_cutoff = fpr[idx]
    fnr_at_fpr_cutoff = fnr[idx]

    if fpr_at_fpr_cutoff > 0.1:
        fnr_at_fpr_cutoff = 1.0

    return fnr_at_fpr_cutoff


def compute_auroc(out_scores, in_scores):
    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    auroc = roc_auc_score(y_true=y_true, y_score=y_score)

    return auroc


# test function
def test(epoch):
    """
    tests current model

    """
    print('validation and testing...')

    net.eval()

    # in-distribution performance
    print("computing over test in-distribution data...\n")
    with torch.no_grad():
        accuracies = []
        OOD_scores_P0 = []
        for data, target in test_loader_in:
            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
            if args.score in ["woods_nn"]:
                # classification accuracy
                output_classification = output[:len(data), :num_classes]
                pred = output_classification.data.max(1)[1]
                accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))
                # OOD scores
                np_in = to_np(output[:, num_classes])
                np_in_list = list(np_in)
                OOD_scores_P0.extend(np_in_list)

            elif args.score in ['energy', 'OE', 'energy_vos', 'woods', 'scone']:
                # classification accuracy
                pred = output.data.max(1)[1]
                accuracies.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

                if args.score in ['energy', 'energy_vos', 'woods', 'scone']:
                    # OOD scores
                    OOD_scores_P0.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':
                    # OOD scores
                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_P0.extend(list(-np.max(smax, axis=1)))

    # test covariate shift OOD distribution performance
    print("computing over test cor-distribution data...\n")
    with torch.no_grad():
        accuracies_cor = []
        OOD_scores_P_cor = []
        #for data, target in in_loader:
        for data, target in test_loader_cor:
            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
            if args.score in ["woods_nn"]:
                # classification accuracy
                output_classification = output[:len(data), :num_classes]
                pred = output_classification.data.max(1)[1]
                accuracies_cor.append(accuracy_score(list(to_np(pred)), list(to_np(target))))
                # OOD scores
                np_in = to_np(output[:, num_classes])
                np_in_list = list(np_in)
                OOD_scores_P_cor.extend(np_in_list)

            elif args.score in ['energy', 'OE', 'energy_vos', 'woods', 'scone']:
                # classification accuracy
                pred = output.data.max(1)[1]
                accuracies_cor.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

                if args.score in ['energy', 'energy_vos', 'woods', 'scone']:
                    # OOD scores
                    OOD_scores_P_cor.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':
                    # OOD scores
                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_P_cor.extend(list(-np.max(smax, axis=1)))


    # semantic shift OOD distribution performance
    print("computing over test OOD-distribution data...\n")
    with torch.no_grad():
        OOD_scores_P_out = []
        for data, target in test_loader_ood:
            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
            if args.score in ["woods_nn"]:
                # classification accuracy
                output_classification = output[:len(data), :num_classes]
                pred = output_classification.data.max(1)[1]
                # OOD scores
                np_in = to_np(output[:, num_classes])
                np_in_list = list(np_in)
                OOD_scores_P_out.extend(np_in_list)

            elif args.score in ['energy', 'OE', 'energy_vos', 'woods', 'scone']:
                # classification accuracy
                pred = output.data.max(1)[1]

                if args.score in ['energy', 'energy_vos', 'woods', 'scone']:
                    # OOD scores
                    OOD_scores_P_out.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':
                    # OOD scores
                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_P_out.extend(list(-np.max(smax, axis=1)))

    # valid in-distribution performance
    print("computing over valid in-distribution data...\n")
    with torch.no_grad():
        accuracies_val = []
        OOD_scores_val_P0 = []
        for data, target in valid_loader_in:
            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
            if args.score in ["woods_nn"]:
                # classification accuracy
                output_classification = output[:len(data), :num_classes]
                pred = output_classification.data.max(1)[1]
                accuracies_val.append(accuracy_score(list(to_np(pred)), list(to_np(target))))
                # OOD scores
                np_in = to_np(output[:, num_classes])
                np_in_list = list(np_in)
                OOD_scores_val_P0.extend(np_in_list)

            elif args.score in ['energy', 'OE', 'energy_vos', 'woods', 'scone']:
                # classification accuracy
                pred = output.data.max(1)[1]
                accuracies_val.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

                if args.score in ['energy', 'energy_vos', 'woods', 'scone']:
                    # OOD scores
                    OOD_scores_val_P0.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':
                    # OOD scores
                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_val_P0.extend(list(-np.max(smax, axis=1)))

    # valid wild-distribution performance
    print("computing over valid wild-distribution data...\n")
    with torch.no_grad():
        OOD_scores_val_P_wild = []
        for data, target in valid_loader_aux:
            if args.ngpu > 0:
                data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
            if args.score in ["woods_nn"]:
                # classification accuracy
                output_classification = output[:len(data), :num_classes]
                pred = output_classification.data.max(1)[1]
                # OOD scores
                np_in = to_np(output[:, num_classes])
                np_in_list = list(np_in)
                OOD_scores_val_P_wild.extend(np_in_list)

            elif args.score in ['energy', 'OE', 'energy_vos', 'woods', 'scone']:
                # classification accuracy
                pred = output.data.max(1)[1]

                if args.score in ['energy', 'energy_vos', 'woods', 'scone']:
                    # OOD scores
                    OOD_scores_val_P_wild.extend(list(-to_np((args.T * torch.logsumexp(output / args.T, dim=1)))))

                elif args.score == 'OE':
                    # OOD scores
                    smax = to_np(F.softmax(output, dim=1))
                    OOD_scores_val_P_wild.extend(list(-np.max(smax, axis=1)))

    in_scores = np.array(OOD_scores_P0)

    in_scores.sort()
    threshold_idx = int(len(in_scores)*0.95)
    threshold = in_scores[threshold_idx]

    val_in = np.array(OOD_scores_val_P0)
    val_wild = np.array(OOD_scores_val_P_wild)

    val_wild_total = len(val_wild)
    val_wild_class_as_in = np.sum(val_wild < threshold)

    print("\n validation wild total {}".format(val_wild_total))
    print("\n validation wild classify as in {}".format(val_wild_class_as_in))

    # compute FPR95 and accuracy
    fpr95 = compute_fnr(np.array(OOD_scores_P_out), np.array(OOD_scores_P0))
    auroc = compute_auroc(np.array(OOD_scores_P_out), np.array(OOD_scores_P0))
    
    acc = sum(accuracies) / len(accuracies)

    acc_cor = sum(accuracies_cor) / len(accuracies_cor)

    # store and print result
    state['fpr95_test'].append(fpr95)
    state['auroc_test'].append(auroc)
    state['test_accuracy'].append(acc)
    state['test_accuracy_cor'].append(acc_cor)
    state['OOD_scores_P0_test'].append(OOD_scores_P0)
    state['OOD_scores_Ptest'].append(OOD_scores_P_out)
    state['val_wild_total'].append(val_wild_total)
    state['val_wild_class_as_in'].append(val_wild_class_as_in)

    wandb.log({"fpr95_test": fpr95,
            "auroc_test": auroc,
            "test_accuracy": acc,
            "test_accuracy_cor":acc_cor,
            "val_wild_total": val_wild_total,
            "val_wild_class_as_in": val_wild_class_as_in,
            'epoch':epoch})

    print("\n fpr95_test {}".format(state['fpr95_test']))
    print("\n auroc_test {}".format(state['auroc_test']))
    print("test_accuracy {} \n".format(state['test_accuracy']))
    print("test_accuracy_cor {} \n".format(state['test_accuracy_cor']))
    print("val_wild_total {} \n".format(state['val_wild_total']))
    print("val_wild_class_as_in {} \n".format(state['val_wild_class_as_in']))


def evaluate_classification_loss_training():
    '''
    evaluate classification loss on training dataset
    '''

    net.eval()
    losses = []
    for in_set in train_loader_in:
        data = in_set[0]
        target = in_set[1]

        if args.ngpu > 0:
            data, target = data.cuda(), target.cuda()
        # forward
        x = net(data)

        # in-distribution classification accuracy
        x_classification = x[:, :num_classes]
        loss_ce = F.cross_entropy(x_classification, target, reduction='none')

        losses.extend(list(to_np(loss_ce)))

    avg_loss = np.mean(np.array(losses))
    print("average loss fr classification {}".format(avg_loss))

    return avg_loss


def evaluate_energy_logistic_loss():
    '''
    evaluate energy logistic loss on training dataset
    '''

    net.eval()
    sigmoid_energy_losses = []
    logistic_energy_losses = []
    ce_losses = []
    for in_set in train_loader_in:
        data = in_set[0]
        target = in_set[1]

        if args.ngpu > 0:
            data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # compute energies
        Ec_in = torch.logsumexp(x, dim=1)

        # compute labels
        binary_labels_1 = torch.ones(len(data)).cuda()

        # compute in distribution logistic losses
        logistic_loss_energy_in = F.binary_cross_entropy_with_logits(logistic_regression(
            Ec_in.unsqueeze(1)).squeeze(), binary_labels_1, reduction='none')

        logistic_energy_losses.extend(list(to_np(logistic_loss_energy_in)))

        # compute in distribution sigmoid losses
        sigmoid_loss_energy_in = torch.sigmoid(logistic_regression(
            Ec_in.unsqueeze(1)).squeeze())

        sigmoid_energy_losses.extend(list(to_np(sigmoid_loss_energy_in)))

        # in-distribution classification losses
        x_classification = x[:, :num_classes]
        loss_ce = F.cross_entropy(x_classification, target, reduction='none')

        ce_losses.extend(list(to_np(loss_ce)))

    avg_sigmoid_energy_losses = np.mean(np.array(sigmoid_energy_losses))
    print("average sigmoid in distribution energy loss {}".format(avg_sigmoid_energy_losses))

    avg_logistic_energy_losses = np.mean(np.array(logistic_energy_losses))
    print("average in distribution energy loss {}".format(avg_logistic_energy_losses))

    avg_ce_loss = np.mean(np.array(ce_losses))
    print("average loss fr classification {}".format(avg_ce_loss))

    return avg_sigmoid_energy_losses, avg_logistic_energy_losses, avg_ce_loss

print('Beginning Training\n')

#compute training loss for scone/woods methods
if args.score in [ 'woods_nn', 'woods', 'scone']:
    full_train_loss = evaluate_classification_loss_training()

###################################################################
# Main loop
###################################################################

import time

for epoch in range(0, args.epochs):
    print('epoch', epoch + 1, '/', args.epochs)
    state['epoch'] = epoch

    begin_epoch = time.time()

    train(epoch)

    test(epoch)

    scheduler.step()

state['best_epoch_valid'] = epoch

state['test_fpr95_at_best'] = state['fpr95_test'][-1]
state['test_auroc_at_best'] = state['auroc_test'][-1]
state['test_accuracy_at_best'] = state['test_accuracy'][-1]
state['test_accuracy_cor_at_best'] = state['test_accuracy_cor'][-1]
state['val_wild_total_at_best'] = state['val_wild_total'][-1]
state['val_wild_class_as_in_at_best'] = state['val_wild_class_as_in'][-1]

print('best epoch = {}'.format(state['best_epoch_valid']))

wandb.log({"best_epoch_valid": state['best_epoch_valid'],
            "test_fpr95_at_best": state['test_fpr95_at_best'],
            "test_auroc_at_best": state['test_auroc_at_best'],
            "test_accuracy_at_best": state['test_accuracy_at_best'],
            "test_accuracy_cor_at_best": state['test_accuracy_cor_at_best'],
            "val_wild_total_at_best": state['val_wild_total_at_best'],
            "val_wild_class_as_in_at_best": state['val_wild_class_as_in_at_best']
            })

# save model checkpoint
#args.checkpoints_dir = './checkpoints/save/'
if args.checkpoints_dir != '':
    model_checkpoint_dir = os.path.join(args.checkpoints_dir,
                                        args.dataset,
                                        args.aux_out_dataset,
                                        args.score)
    if not os.path.exists(model_checkpoint_dir):
        os.makedirs(model_checkpoint_dir, exist_ok=True)
    model_filename = '{}_epoch_{}.pt'.format(method_data_name, epoch)
    model_path = os.path.join(model_checkpoint_dir, model_filename)
    print('saving model to {}'.format(model_path))
    torch.save(net.state_dict(), model_path)

# save results to .pkl file
results_dir = os.path.join(args.results_dir,
                            args.dataset,
                            args.aux_out_dataset,
                            args.score)
if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)
results_filename = '{}.pkl'.format(method_data_name)
results_path = os.path.join(results_dir, results_filename)
with open(results_path, 'wb') as f:
    print('saving results to', results_path)
    pickle.dump(state, f)




