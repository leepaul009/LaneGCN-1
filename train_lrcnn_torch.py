# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib.metadata import distribution
import os

os.umask(0)
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from numbers import Number

from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
# import horovod.torch as hvd
torch.set_num_threads(6)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from utils import Logger, load_pretrain, load_pretrain_dist
from utils import gpu, to_long,  Optimizer, StepLR

from mpi4py import MPI
from lib import comm
import copy


mpi_comm = MPI.COMM_WORLD
# hvd.init()
# print("hvd.local_rank={}, hvd.rank={}, hvd.size={} ".format( hvd.local_rank(), hvd.rank(), hvd.size() ))
# torch.cuda.set_device(hvd.local_rank())

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument("-m", "--model", default="lanercnn", type=str, metavar="MODEL", help="model name")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--resume", default="", type=str, metavar="RESUME", help="checkpoint path")
parser.add_argument("--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path")
parser.add_argument("--local_rank", type=int, default=0)

def main():
  args = parser.parse_args()

  distributed = True
  device = None
  if distributed:
      device = torch.device('cuda:{}'.format(args.local_rank))
      torch.cuda.set_device(args.local_rank)
      torch.distributed.init_process_group(
          backend="nccl", init_method="env://", rank=args.local_rank, world_size=4)
  else:
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  seed = comm.get_rank()
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  model = import_module(args.model)
  config, Dataset, collate_fn, net, loss, post_process = model.get_model_for_torch_dist()

  save_dir = config["save_dir"]
  log = os.path.join(save_dir, "log")

  if comm.is_main_process():
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    sys.stdout = Logger(log)

  dataset = Dataset(config["train_split"], config, train=True)

  if distributed:
    train_sampler = DistributedSampler(
      dataset, 
      shuffle=True)
  else:
    train_sampler = torch.utils.data.RandomSampler(dataset)

  train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    pin_memory=False,
    num_workers=config["workers"],
    sampler=train_sampler,
    collate_fn=collate_fn,
    worker_init_fn=worker_init_fn,
    )


  val_loader = None
  # dataset = Dataset(config["val_split"], config, train=False)
  # val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
  # val_loader = DataLoader(
  #   dataset,
  #   batch_size=config["val_batch_size"],
  #   num_workers=config["val_workers"],
  #   sampler=val_sampler,
  #   collate_fn=collate_fn,
  #   pin_memory=True,
  # )
  if distributed:
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DistributedDataParallel(
            net, 
            find_unused_parameters=True,
            device_ids=[args.local_rank], 
            output_device=args.local_rank)
  
  opt = Optimizer(net.parameters(), config)
  
  if args.resume or args.weight:
    ckpt_path = args.resume or args.weight
    if not os.path.isabs(ckpt_path):
      ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    # load_pretrain(net, ckpt["state_dict"])
    load_pretrain_dist(net, distributed, ckpt["state_dict"])
    if args.resume:
      config["epoch"] = ckpt["epoch"]
      opt.load_state_dict(ckpt["opt_state"])

  epoch = config["epoch"]
  remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
  print("remaining_epochs = {} ".format(remaining_epochs))
  for i in range(remaining_epochs):
    train(epoch + i, config, train_loader, net, loss, post_process, opt, val_loader)


def worker_init_fn(pid):
    # np_seed = hvd.rank() * 1024 + int(pid)
    np_seed = comm.get_rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def train(epoch, config, train_loader, net, loss, post_process, opt, val_loader=None):
  train_loader.sampler.set_epoch(int(epoch))
  net.train()

  num_batches = len(train_loader)
  epoch_per_batch = 1.0 / num_batches
  save_iters = int(np.ceil(config["save_freq"] * num_batches))
  display_iters = int(
      config["display_iters"] / (comm.get_world_size() * config["batch_size"])
  )
  val_iters = int(config["val_iters"] / (comm.get_world_size() * config["batch_size"]))
  print("display_iters={}, val_iters={}".format(display_iters, val_iters))
  
  start_time = time.time()
  metrics = dict()

  for i, data in enumerate(train_loader):
      epoch += epoch_per_batch
      data = dict(data)

      for batch_i in range(len(data["gt_preds"])):
          orig, rot = data["orig"][batch_i], data["rot"][batch_i]
          data["gt_preds"][batch_i] = torch.matmul(
            (data["gt_preds"][batch_i].view(-1, 30, 2) - orig.view(1, 1, -1)), 
            rot.transpose(1, 0) )

      output   = net(data)
      loss_out = loss(output, data)
      post_out = post_process(output, data, loss_out)
      del data
      post_process.append(metrics, loss_out, post_out)

      opt.zero_grad()
      loss_out["loss"].backward()
      lr = opt.step(epoch)

      # if i % 100 == 0:
      #   print("loss: {}".format( 
      #     loss_out["loss"].detach().cpu().item() ))

      num_iters = int(np.round(epoch * num_batches))
      # if hvd.rank() == 0 and (
      #     num_iters % save_iters == 0 or epoch >= config["num_epochs"]
      # ):
      #     save_ckpt(net, opt, config["save_dir"], epoch)
      if comm.is_main_process() and (
          num_iters % save_iters == 0 or epoch >= config["num_epochs"]
      ):
          save_ckpt(net, opt, config["save_dir"], epoch)

      # if num_iters % display_iters == 0:
      if i % 100 == 0:
          dt = time.time() - start_time
          metrics = sync(metrics)
          # if hvd.rank() == 0:
          if comm.is_main_process():
              post_process.display(metrics, dt, epoch, lr)
          start_time = time.time()
          metrics = dict()
      
      # if i % 1000 == 0:
      #     val(config, val_loader, net, loss, post_process, epoch)
      # if num_iters % val_iters == 0:
      #     val(config, val_loader, net, loss, post_process, epoch)

      # if epoch >= config["num_epochs"]:
      #     val(config, val_loader, net, loss, post_process, epoch)
      #     return

# def val(config, data_loader, net, loss, post_process, epoch):
#   net.eval()
  
#   start_time = time.time()
#   metrics = dict()
#   for i, data in enumerate(data_loader):
#     data = dict(data)
#     for batch_i in range(len(data["gt_preds"])):
#       orig, rot = data["orig"][batch_i], data["rot"][batch_i]
#       data["gt_preds"][batch_i] = torch.matmul(
#         (data["gt_preds"][batch_i].view(-1, 30, 2) - orig.view(1, 1, -1)), 
#         rot.transpose(1, 0) )

#     with torch.no_grad():
#       output = net(data)
#       loss_out = loss(output, data)
#       post_out = post_process(output, data, loss_out)
#       post_process.append(metrics, loss_out, post_out)
  
#   dt = time.time() - start_time
#   metrics = sync(metrics)
#   if hvd.rank() == 0:
#       post_process.display(metrics, dt, epoch)
#   net.train()


def save_ckpt(net, opt, save_dir, epoch):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  
  state_dict = net.module.state_dict() # for distributed
  for key in state_dict.keys():
    state_dict[key] = state_dict[key].cpu()

  save_name = "%3.3f.ckpt" % epoch
  torch.save(
    {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
    os.path.join(save_dir, save_name),
  )

def sync(data):
    # data_list = comm.allgather(data)
    data_list = mpi_comm.allgather(data)
    data = dict()
    for key in data_list[0]:
        if isinstance(data_list[0][key], list):
            data[key] = []
        else:
            data[key] = 0
        for i in range(len(data_list)):
            data[key] += data_list[i][key]
    return data

if __name__ == "__main__":
  main()