"""
This file is the template for the boilerplate of train/test of a DNN

There are a lot of parameter which can be specified to modify the behaviour
and they should be used instead of hard-coding stuff.

@authors: Vinaychandran Pondenkandath , Michele Alberti
"""

# Utils
import argparse
import json
import logging
import os
import time

# Tensor board
import tensorboardX
# Torch related stuff
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

# DeepDIVA
import dataset
import model as models
from util.misc import AverageMeter, accuracy

###############################################################################
# Argument Parser

# Training Settings
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Template for training CNN on CIFAR')

# General Options
parser.add_argument('--experiment-name',
                    help='provide a meaningful and descriptive name to this run',
                    default=None, type=str)

# Data Options
parser.add_argument('--dataset',
                    help='one of {CIFAR10, CIFAR100}', default='CIFAR10')
parser.add_argument('--log-dir',
                    help='where to save logs', default='./data/')

# Training Options
parser.add_argument('--model',
                    help='which model to use for training',
                    type=str, default='CNN_Basic')
parser.add_argument('--lr',
                    help='learning rate to be used for training',
                    type=float, default=0.001)
parser.add_argument('--optimizer',
                    help='optimizer to be used for training. {Adam, SGD}',
                    default='Adam')
parser.add_argument('--batch-size',
                    help='input batch size for training',
                    type=int, default=64)
parser.add_argument('--test-batch-size',
                    help='input batch size for testing',
                    type=int, default=64)
parser.add_argument('--epochs',
                    help='how many epochs to train',
                    type=int, default=100)

# System Options
parser.add_argument('--gpu-id',
                    default=None,
                    help='which GPUs to use for training (use all by default)')
parser.add_argument('--no-cuda',
                    default=False, action='store_true', help='run on CPU')
parser.add_argument('--log-interval',
                    default=10, type=int,
                    help='print loss/accuracy every N batches')
parser.add_argument('-j', '--workers',
                    default=4, type=int,
                    help='workers used for train/val loaders')
args = parser.parse_args()

###############################################################################
# Setup Logging
basename = args.log_dir

# Experiment name override
if args.experiment_name is None:
    vars(args)['experiment_name'] = input("Experiment name:")

experiment_name = args.experiment_name
log_folder = os.path.join(basename, experiment_name,
                          '{}'.format(time.strftime('%y-%m-%d-%Hh-%Mm-%Ss')))
logfile = 'logs.txt'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

logging.basicConfig(
    format='%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s',
    filename=os.path.join(log_folder, logfile),
    level=logging.INFO)
logging.info(
    'Set up logging. Log file: {}'.format(os.path.join(log_folder, logfile)))

# Save args to logs_folder
logging.info('Arguments saved to: {}'.format(os.path.join(log_folder, 'args.txt')))
with open(os.path.join(log_folder, 'args.txt'), 'w') as f:
    f.write(json.dumps(vars(args)))

# Define Tensorboard SummaryWriter
logging.info('Initialize Tensorboard SummaryWriter')
writer = tensorboardX.SummaryWriter(log_dir=log_folder)

# Set visible GPUs
if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


###############################################################################
def main():
    """
    This is the main routine where train() and validate() are called.
    :return:
        None
    """

    # Loading dataset
    logging.info('Initalizing dataset {}'.format(args.dataset))

    model_expected_input_size = (32, 32)
    logging.info('Model {} expects input size of {}'.format(args.model,
                                                            model_expected_input_size))

    train_ds = dataset.__dict__[args.dataset](root='data/datasets',
                                              train=True,
                                              download=True)

    train_ds.transform = transforms.Compose([
        transforms.Scale(model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_ds.mean, std=train_ds.std)
    ])

    test_ds = dataset.__dict__[args.dataset](root='data/datasets',
                                             train=False,
                                             download=True)

    test_ds.transform = transforms.Compose([
        transforms.Scale(model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_ds.mean, std=train_ds.std)
    ])

    # Setup dataloaders
    logging.info('Set up dataloaders')
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=args.batch_size,
                                              num_workers=args.workers,
                                              pin_memory=True)

    # Initialize the model
    logging.info('Initialize model')
    model = models.__dict__[args.model](train_ds.num_classes)
    optimizer = torch.optim.__dict__[args.optimizer](model.parameters(),
                                                     args.lr)
    criterion = nn.CrossEntropyLoss()

    # Transfer model to GPU (if desired)
    if not args.no_cuda:
        logging.info('Transfer model to GPU')
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # Begin training
    logging.info('Begin training')
    best_acc = 0
    best_model = model.state_dict()
    for i in range(args.epochs):
        train(train_loader, model, criterion, optimizer, i)
        best_acc, best_model = validate(test_loader, model, criterion, i, best_acc, best_model)

    logging.info('* Best Acc:{0}'.format(str(best_acc)))
    logging.info('Saving best model')

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), os.path.join(log_folder, 'best_model.pt'))

    logging.info('Training completed')

    writer.close()


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Training routine
    :param train_loader:    torch.utils.data.DataLoader
        The dataloader of the train set
    :param model:           torch.nn.module
        The network model being used
    :param criterion:       torch.nn.loss
        The loss function used to compute the loss of the model
    :param optimizer:       torch.optim
        The optimizer used to perform the weight update
    :param epoch:
        Number of the epoch (mainly for logging purposes)
    :return:
        None
    """

    # Init the counters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to train mode (turn on dropout & stuff)
    model.train()

    # Iterate over whole training set
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not args.no_cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Compute output
        output = model(input_var)

        # Compute and record the loss
        loss = criterion(output, target_var)
        losses.update(loss.data[0], input.size(0))

        # Compute and record the accuracy
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # Add loss and accuracy to Tensorboard
        writer.add_scalar('train/mb_loss', loss.data[0], epoch * len(train_loader) + i)
        writer.add_scalar('train/mb_accuracy', acc1.cpu().numpy(), epoch * len(train_loader) + i)

        # Reset gradient
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Perform a step by updating the weights
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log to console
        if i % args.log_interval == 0:
            logging.info('Epoch [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    # Logging the epoch-wise accuracy
    writer.add_scalar('train/accuracy', top1.avg, epoch)

    return


def validate(val_loader, model, criterion, epoch, best_acc, best_model):
    """
    The validation routine
    :param val_loader:    torch.utils.data.DataLoader
        The dataloader of the train set
    :param model:           torch.nn.module
        The network model being used
    :param criterion:       torch.nn.loss
        The loss function used to compute the loss of the model
    :param epoch:
        Number of the epoch (mainly for logging purposes)
    :return:
        None
    """

    # Init the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole validation set
    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        # Moving data to GPU
        if not args.no_cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # Compute output
        output = model(input_var)

        # Compute and record the loss
        loss = criterion(output, target_var)
        losses.update(loss.data[0], input.size(0))

        # Compute and record the accuracy
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # Add loss and accuracy to Tensorboard
        writer.add_scalar('val/mb_loss', loss.data[0],
                          epoch * len(val_loader) + i)
        writer.add_scalar('val/mb_accuracy', acc1.cpu().numpy(),
                          epoch * len(val_loader) + i)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            logging.info('Epoch [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    # Logging the epoch-wise accuracy
    writer.add_scalar('val/accuracy', top1.avg, epoch - 1)

    logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    logging.info(' * Alpha {alpha} Grad {grad}'
          .format(alpha=float(model.module.alpha.data.mean()), grad=model.module.alpha._grad.data.mean() if model.module.alpha._grad is not None else 0))

    if top1.avg >= best_acc:
        best_acc = top1.avg
        best_model = model.state_dict()
    return best_acc, best_model


if __name__ == "__main__":
    # Set up logging to console
    fmtr = logging.Formatter(fmt='%(funcName)s %(levelname)s: %(message)s')
    stderr_handler = logging.StreamHandler()
    stderr_handler.formatter = fmtr
    logging.getLogger().addHandler(stderr_handler)
    logging.info('Printing activity to the console')

    main()
