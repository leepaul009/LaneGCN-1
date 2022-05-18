
import enum
import numpy as np
import os
import sys
from fractions import gcd
from numbers import Number
from sklearn.utils import resample

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F

from data_lrcnn import ArgoDataset, collate_fn
from utils import gpu, to_long,  Optimizer, StepLR

from layers import Conv1d, Res1d, Linear, LinearRes, Null
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import copy

file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)
model_name = os.path.basename(file_path).split(".")[0]

### config ###
config = dict()
"""Train"""
config["display_iters"] = 205942
config["val_iters"] = 205942 * 2
config["save_freq"] = 1.0
config["epoch"] = 0
config["horovod"] = True
config["opt"] = "adam"
config["num_epochs"] = 36
config["lr"] = [1e-3, 1e-4]
config["lr_epochs"] = [32]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])


if "save_dir" not in config:
    config["save_dir"] = os.path.join(
        root_path, "results", model_name
    )

if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "results", config["save_dir"])

config["batch_size"] = 12
config["val_batch_size"] = 12
config["workers"] = 0
config["val_workers"] = config["workers"]


"""Dataset"""
# Raw Dataset
config["train_split"] = os.path.join(
    root_path, "dataset/train/data"
)
config["val_split"] = os.path.join(root_path, "dataset/val/data")
config["test_split"] = os.path.join(root_path, "dataset/test_obs/data")

# Preprocessed Dataset
config["preprocess"] = True # whether use preprocess or not
config["preprocess_train"] = os.path.join(
    root_path, "dataset","preprocess", "train_crs_dist6_angle90.p"
)
config["preprocess_val"] = os.path.join(
    root_path,"dataset", "preprocess", "val_crs_dist6_angle90.p"
)
config['preprocess_test'] = os.path.join(
    root_path, "dataset",'preprocess', 'test_test.p')

"""Model"""
config["rot_aug"] = False
config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
config["num_scales"] = 6
config["n_actor"] = 128
config["n_map"] = 128
config["actor2map_dist"] = 7.0
config["map2actor_dist"] = 6.0
config["actor2actor_dist"] = 100.0
config["pred_size"] = 30
config["pred_step"] = 1
config["num_preds"] = config["pred_size"] // config["pred_step"]
config["num_mods"] = 6
config["cls_coef"] = 1.0
config["reg_coef"] = 1.0
config["mgn"] = 0.2
config["cls_th"] = 2.0
config["cls_ignore"] = 0.2
### end of config ###


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.input = LaneInput(config)
        self.roi_net1 = LaneRoI(config, input_dim=config["n_map"])
        self.interactor = Interactor(config)
        self.roi_net2 = LaneRoI(config, input_dim=config["n_map"])

        self.head = PredHead(config)
        # self.pred_net = PredNet(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:        
        # construct map features
        graph = graph_gather(to_long(gpu(data["graph"])))

        # construct laneRoI features
        graphRoI = subgraph_gather(to_long(gpu(data["subgraphs"])))

        # roi_feat = torch.cat(graphRoI["feats"], 0) # [n_rois, 88] concat along batch
        ##
        roi_feat = self.input(graphRoI)
        ##
        roi_feat = self.roi_net1(roi_feat, graphRoI)
        roi_feat = self.interactor(graph, graphRoI, roi_feat)
        roi_feat = self.roi_net2(roi_feat, graphRoI)

        out = dict()
        out['roi_feat'] = self.head(roi_feat)
        out['graphRoI'] = graphRoI
        return out


def subgraph_gather(subgraphs_in_batch):
    graph = dict()

    batch_size = len(subgraphs_in_batch)
    node_idcs = []
    count = 0
    counts_per_agt = []
    num_atgs_per_batch = []
    spans = []
    start = 0
    for batch_idx in range(batch_size):
        subgraphs = subgraphs_in_batch[batch_idx]
        num_atgs  = len(subgraphs)
        num_atgs_per_batch.append(num_atgs)
        num_nodes_this_batch = 0
        for atg_i in range(num_atgs):
            counts_per_agt.append(count)
            num_nodes = len( subgraphs[atg_i]['feats'] )
            idcs = torch.arange(count, count + num_nodes).to(
                subgraphs[atg_i]['feats'].device
            )
            node_idcs.append(idcs)
            count += num_nodes
            num_nodes_this_batch += num_nodes
        spans.append([start, start + num_nodes_this_batch])
        start += num_nodes_this_batch
  
    graph["num_nodes"]   = count
    graph["node_idcs"]   = torch.cat(node_idcs, 0)
    graph["counts"]      = counts_per_agt # offset per laneRoi
    graph["batch_spans"] = spans  # batch span
    graph["num_atgs_per_batch"] = num_atgs_per_batch

    temp = copy.deepcopy(counts_per_agt)
    temp.append(count)
    graph["agt_spans"] = [[temp[i], temp[i+1]] for i in range(len(temp)-1)]

    idx_roi = 0
    feats, agt_feat, rel_a2m_us, rel_a2m_vs = [], [], [], []
    for batch_i in range(batch_size):
        temp, agt_temp = [], []
        for atg_i in range(num_atgs_per_batch[batch_i]): # merge along agents
            # n_nodes = len(subgraphs_in_batch[batch_i][atg_i]["feats"])
            temp.append( subgraphs_in_batch[batch_i][atg_i]["feats"] )
            agt_temp.append( subgraphs_in_batch[batch_i][atg_i]["agent_feat"].view(1, -1) )

            us = subgraphs_in_batch[batch_i][atg_i]['a2m']['u'] + idx_roi
            vs = subgraphs_in_batch[batch_i][atg_i]['a2m']['v'] + counts_per_agt[idx_roi]
            rel_a2m_us.append(us.long())
            rel_a2m_vs.append(vs.long())
            idx_roi += 1

        # merge laneRoi with a batch
        temp = torch.cat(temp, 0)
        agt_temp = torch.cat(agt_temp, 0)
        feats.append(temp)
        agt_feat.append(agt_temp)
    graph["feats"] = feats # list of tensor, size=batch_size
    graph["agent_feat"] = agt_feat # list of tensor, size=batch_size
    graph["ctrs"]  = [feats[i][:, :2] for i in range(batch_size)]
    graph["pose"]  = [feats[i][:, :4] for i in range(batch_size)]
    graph["a2m"] = {"u": torch.cat(rel_a2m_us, 0), "v": torch.cat(rel_a2m_vs, 0)}
    # graph["a2m"] = to_long(graph["a2m"])
    ## merge edge
    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(6):
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                items_to_cat = []
                idx_batch_agt = 0
                for batch_i in range(batch_size):
                    for atg_i in range(num_atgs_per_batch[batch_i]):
                        if len(subgraphs_in_batch[batch_i][atg_i][k1][i][k2]) > 0:
                            items_to_cat.append( 
                                subgraphs_in_batch[batch_i][atg_i][k1][i][k2] + counts_per_agt[idx_batch_agt]
                            )
                        idx_batch_agt += 1
                if len(items_to_cat) > 0:
                    graph[k1][i][k2] = torch.cat(items_to_cat, 0)
                else:
                    graph[k1][i][k2] = torch.zeros((0,))

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            items_to_cat = []
            idx_batch_agt = 0
            for batch_i in range(batch_size):
                for atg_i in range(num_atgs_per_batch[batch_i]):
                    if len(subgraphs_in_batch[batch_i][atg_i][k1][k2]) > 0:
                        items_to_cat.append(
                            subgraphs_in_batch[batch_i][atg_i][k1][k2] + counts_per_agt[idx_batch_agt]
                        )
                    idx_batch_agt += 1
            if len(items_to_cat) > 0:
                graph[k1][k2] = torch.cat(items_to_cat, 0)
            else:
                graph[k1][k2] = torch.zeros((0,))

    return graph


def graph_gather(graphs):
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["num_nodes"] = [graphs[i]["num_nodes"] for i in range(batch_size)]
    graph["idcs"] = node_idcs # segment id in global(batch)
    graph["counts"] = counts

    graph["ctrs"] = [x["ctrs"] for x in graphs] # segment center point
    graph["pose"] = [torch.cat([x["ctrs"], x["feats"]], -1) for x in graphs]

    for key in ["feats", "turn", "control", "intersect"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0) # feats: [N_batch*N_segs_a_batch,2]

    # apply global id(batch) to pre/suc/left/right
    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])): # scale_num: 6
            graph[k1].append(dict())
            for k2 in ["u", "v"]: # start node, end node
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    return graph


class LaneInput(nn.Module):
    def __init__(self, config):
        super(LaneInput, self).__init__()
        map_dim = config["n_map"]
        norm = "GN"
        ng = 1
        self.map_fc = nn.Sequential(
            nn.Linear(8, map_dim),
            nn.ReLU(inplace=True),
            Linear(map_dim, map_dim, norm=norm, ng=ng)
        )
#         self.agt_fc = nn.Sequential(
#             nn.Linear(80, map_dim),
#             nn.ReLU(inplace=True),
#             Linear(map_dim, map_dim, norm=norm, ng=ng, act=False)
#         )
        self.agt_fc = nn.Linear(80, map_dim, bias=False)
    
        self.fc = Linear(2*map_dim, map_dim, norm=norm, ng=ng)
        self.bn = nn.GroupNorm(gcd(ng, map_dim), map_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, graph):
        map_feats = torch.cat(graph["feats"], 0) # [nodes, 8]
        agt_feats = torch.cat(graph["agent_feat"], 0) # [agts, 80]

        map_feats = self.map_fc(map_feats) # [nodes, 128], fc+relu+fc+norm+relu

        tmp_feats = torch.zeros_like(map_feats) # [nodes, 128]
        tmp_feats.index_add(
            0,
            graph["a2m"]["v"],
            self.agt_fc( agt_feats[ graph["a2m"]["u"] ] ), # fc
        )
        tmp_feats = self.bn(tmp_feats)
        tmp_feats = self.relu(tmp_feats)
        
        map_feats = torch.cat([map_feats, tmp_feats], -1)
        
        map_feats = self.fc(map_feats) # fc+norm+relu
        return map_feats


class LaneRoI(nn.Module):
    def __init__(self, config, input_dim):
        super(LaneRoI, self).__init__()
        self.config = config
        map_dim = config["n_map"] # 128
        norm = "GN"
        ng = 1

        self.input = Linear(input_dim, map_dim, norm=norm, ng=ng, act=True)

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []
    
        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, map_dim), map_dim))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(map_dim, map_dim, norm=norm, ng=ng, act=False))
                    # fuse[key].append(nn.Linear(map_dim, map_dim, bias=False))
                else: # ctr, pre/suc, left/right
                    fuse[key].append(nn.Linear(map_dim, map_dim, bias=False))
    
        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat, graph):
        feat = self.input(feat) # FC+GN+ReLU
        identity = feat
        for i in range(4):
            temp = self.fuse["ctr"][i](feat) # FC
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1  = key[:3]
                    hop = int(key[3:])
                    if len(graph[k1][hop]["u"]) > 0:
                        temp.index_add_(
                            0,
                            graph[k1][hop]["u"],
                            self.fuse[key][i]( feat[ graph[k1][hop]["v"] ] ), # FC
                        )
            if len(graph["left"]["u"]) > 0:
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i]( feat[ graph["left"]["v"] ] ), # FC
                )
            if len(graph["right"]["u"]) > 0:
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i]( feat[ graph["right"]["v"] ] ), # FC
                )
            feat = self.fuse["norm"][i](temp) # GN
            feat = self.relu(feat)            # ReLU
            feat = self.fuse["ctr2"][i](feat) # FC+GN
            # feat = self.relu(feat)

            feat += identity
            # TODO：use layer norm to replace relu, then do short-cut
            feat = self.relu(feat) # feat = self.layer_norm(feat)

            ## short-cut：
#             if i == 1 or i == 3:
#                 agg_feat, _ = torch.max(identity, dim=0)
#                 feat = feat + agg_feat

            identity = feat
        return feat


class LanePooling(nn.Module):
    # ex. n_mnode:128, n_agent:128
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(LanePooling, self).__init__()
        norm = "GN"
        ng = 1
        in_dim, mid_dim, out_dim = 128, 128, 128
        # mid_dim = 128
        self.input = nn.Linear(in_dim, mid_dim, bias=False)
        self.relpose = nn.Sequential(
            nn.Linear(4, in_dim),
            nn.ReLU(inplace=True),
        )
        self.ctx = nn.Sequential(
            Linear(in_dim * 2, mid_dim, norm=norm, ng=ng),
            nn.Linear(mid_dim, mid_dim, bias=False),
        )
        # self.mlp = Linear(mid_dim, out_dim, norm=norm, ng=ng, act=False)
        self.mlp = nn.Sequential(
            # nn.Linear(mid_dim, mid_dim),
            # nn.ReLU(inplace=True),
            Linear(mid_dim, mid_dim, norm=norm, ng=ng),
            Linear(mid_dim, out_dim, norm=norm, ng=ng, act=False),
        )
        self.norm = nn.GroupNorm(gcd(ng, 128), 128)
        self.relu = nn.ReLU(inplace=True)
    
    ## input:
    ##  context_feat: roi_lane feature
    ##  target_feat:  g_graph feature
    def forward(self, context_feat, context_graph, target_feat, target_graph, dist_th=2.0, g2r=False):
        # list[tensor] per batch
        context_ctrs_batch, target_ctrs_batch = context_graph['ctrs'], target_graph['ctrs']
        # tensor merged along batch
        context_pose_batch, target_pose_batch = context_graph['pose'], target_graph['pose']
        batch_size = len(target_graph["ctrs"])

        ## compute distance between centers of context and target graph
        hi, wi = [], []
        hi_count, wi_count = 0, 0

        for batch_i in range(batch_size):
            context_ctrs = context_ctrs_batch[batch_i]
            target_ctrs  = target_ctrs_batch[batch_i]
            dist = context_ctrs.view(-1, 1, 2) - target_ctrs.view(1, -1, 2) # dist, shape=[c_nodes, t_nodes, 2]
            dist = torch.sqrt((dist ** 2).sum(2)) # dist, shape=[c_nodes, t_nodes]
            mask = dist <= dist_th

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(context_ctrs)
            wi_count += len(target_ctrs)
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        context_pose_batch = torch.cat(context_pose_batch, 0)
        target_pose_batch  = torch.cat(target_pose_batch, 0)
        dist_feat = context_pose_batch[hi] - target_pose_batch[wi] # [n_rel, 4]
        dist_feat = self.relpose(dist_feat) # fc + relu [n_rel, 128]
        
        ## compute context feature
        ctx = torch.cat([context_feat[hi], dist_feat], -1) # [n_rel, 128 + 128]
        ctx = self.ctx(ctx) # M_a: fc + norm + relu + fc
        
        # if g2r:
        identity = target_feat
        # do not add this step will make input be changed
        # otherwise bp update target_feat that is also input of another network
        target_feat = self.input(target_feat) # fc
        
        
        target_feat.index_add_(0, wi, ctx) # add context feature to target feature
        target_feat = self.norm(target_feat)
        target_feat = self.relu(target_feat)

        target_feat = self.mlp(target_feat) # M_b: fc + relu + fc + norm
        # if g2r:
        target_feat += identity
        target_feat = self.relu(target_feat)

        return target_feat


class GlobalGraphNet(nn.Module):
    """
    Map Graph feature extractor with LaneGraphCNN
    """
    def __init__(self, config):
        super(GlobalGraphNet, self).__init__()
        self.config = config
        n_map = config["n_map"] # 128
        norm = "GN"
        ng = 1

        ## keys: ctr, norm, ctr2, left, right, pre0~pre5, suc0~suc5
        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]): # 6
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat, graph):
        if (
            len(graph["feats"]) == 0
            or len(graph["pre"][-1]["u"]) == 0
            or len(graph["suc"][-1]["u"]) == 0
        ): # either zero node or zero edge
            temp = graph["feats"]
            return (
                temp.new().resize_(0),
                # [temp.new().long().resize_(0) for x in graph["node_idcs"]],
                # temp.new().resize_(0),
            )

        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])): # 4
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                # learn from pre/suc_0~5
                if key.startswith("pre") or key.startswith("suc"): 
                    k1 = key[:3] # pre or suc
                    k2 = int(key[3:]) # 0~5
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"], # ex. pre_1_u(seg ids)
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat


class Interactor(nn.Module):
    """
    Map Graph feature extractor with LaneGraphCNN
    """
    def __init__(self, config):
        super(Interactor, self).__init__()
        self.config = config
        n_map = config["n_map"] # 128
        norm = "GN"
        ng = 1

        self.input = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )
        self.seg = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )
        self.relu = nn.ReLU(inplace=True)

        self.roi2graph        = LanePooling(in_dim=128, out_dim=128)
        self.global_graph_net = GlobalGraphNet(config)
        self.graph2roi        = LanePooling(in_dim=128, out_dim=128)

    def forward(self, graph, subgraph, roi_feat):
        
        ## graph input -> graph feat
        '''
        ctrs = torch.cat(graph["ctrs"], 0)
        graph_feat = self.input(ctrs)           # [g_nodes, 2] -> [g_nodes, 128]
        graph_feat += self.seg(graph["feats"])  # [g_nodes, 2] -> [g_nodes, 128]
        graph_feat = self.relu(graph_feat)
        '''
        
        # if True:
        graph_feat = torch.zeros(( len(graph['feats']), 128 ), dtype=torch.float, device=roi_feat.device)
        
        ## lane pooling: roi feat -> graph feat
        graph_feat = self.roi2graph(roi_feat, subgraph, graph_feat, graph, g2r=False)

        ## map net: graph feat -> graph feat
        graph_feat = self.global_graph_net(graph_feat, graph)
        
        ## lane pooling: graph feat -> roi feat
        roi_feat = self.graph2roi(graph_feat, graph, roi_feat, subgraph, g2r=True)

        return roi_feat


class PredHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        norm = "GN"
        ng = 1
        self.k = config["num_mods"] # 6
        n_actor = config["n_actor"] # 128
        self.pred = nn.Sequential(
            # nn.Linear(n_actor, n_actor),
            # nn.ReLU(inplace=True),
            # Linear(n_actor, 5, norm=norm, ng=ng),
            Linear(n_actor, n_actor, norm=norm, ng=ng),
            nn.Linear(n_actor, 5),
        )
    
    def forward(self, roi_feat):
        out = self.pred(roi_feat) # [roi_nodes, 128] => [roi_nodes, 5]
        return out


class RoiLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_mods = config["num_mods"] # 6
        self.bce_loss = nn.BCELoss()
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self,
                data,
                out: Dict,
                gt_preds: List[Tensor], 
                has_preds: List[Tensor]):
        
        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"] # 6, 30
        preds      = out["roi_feat"] # [?, 5]
        subgraph   = out['graphRoI']
        device     = preds.device
        batch_size = len(gt_preds)

        valid_agent_ids = to_long(gpu(data['valid_agent_ids']))
        gt_preds  = [x[ids].view(-1,30,2) for ids, x in zip( valid_agent_ids, gt_preds )]
        has_preds = [x[ids].view(-1,30)   for ids, x in zip( valid_agent_ids, has_preds )]
        agt_ctrs  = [x[ids].view(-1,2)    for ids, x in zip( valid_agent_ids, gpu(data['ctrs']) )]
        agt_dirs  = [x[ids].view(-1,20,3) for ids, x in zip( valid_agent_ids, gpu(data['feats']) )]
        # NAs : num of agents (sum along batch)
        agt_ctrs  = torch.cat(agt_ctrs, 0)  # [NAs, 2]
        agt_dirs  = torch.cat(agt_dirs, 0) # [NAs, 20, 3]
        agt_poses = torch.cat([agt_ctrs, agt_dirs[:, -1, :2]], -1) # [NAs, 4]
        # agt_vel      = torch.sqrt(( agt_dirs[:, -1, :2]**2 ).sum(-1)) / 0.1
        # agt_gt_preds = torch.cat(data['gt_preds'], 0).to(device)
        # agt_s        = agt_gt_preds[:, 1:] - agt_gt_preds[:, :-1]
        # agt_s        = torch.sum( torch.sqrt((agt_s**2).sum(-1)), -1)
        # agt_acc      = 2 * (agt_s - agt_vel * 3) / 9

        ## to local coordinate
        # rots, origs = gpu(data["rot"]), gpu(data["orig"])
        # for batch_i in range(batch_size):
        #     orig, rot = origs[batch_i], rots[batch_i]
        #     gt_preds[batch_i] = torch.matmul( (gt_preds[batch_i] - orig.view(1, 1, -1)), rot.transpose(1, 0) )
        
        gt_preds  = torch.cat(gt_preds, 0)
        has_preds = torch.cat(has_preds, 0)
        # gt_preds_orig  = copy.deepcopy(gt_preds)  # [NAs, 30, 2]
        # has_preds_orig = copy.deepcopy(has_preds) # [NAs, 30]
        # gt_preds  = gt_preds.unsqueeze(1).repeat(1, num_mods, 1, 1) # [NAs, 6, 30, 2]
        # has_preds = has_preds.unsqueeze(1).repeat(1, num_mods, 1)   # [NAs, 6, 30]


        loss_out = dict()
        anchors  = torch.cat(subgraph["pose"], 0)
        preds_list, logics_list  = [], []
        stage_one_loss, num_stage_one = 0, 0
        for agt_i, span in enumerate(subgraph["agt_spans"]): # per agent(or lane_roi) = NAs
            anchor  = anchors[span[0] : span[1]] # [nodes_per_roi, 4]
            pred    = preds[span[0] : span[1]]   # [nodes_per_roi, 5]
            logics  = pred[:, 0]  # [nodes_per_roi,]
            poses   = pred[:, 1:] # [nodes_per_roi, 4]

            pred_goals = anchor + poses
            
            # dist = anchor[:, :2] - gt_preds[agt_i, -1].reshape(1,-1) #  [nodes_per_roi, 2] - [1,2]
            # dist = torch.sqrt(dist**2).sum(-1) # [nodes_per_roi,]
            # matched_node_id = torch.argmin(dist).reshape(1) # get best matched node
            # stage_one_loss += F.nll_loss(logics.unsqueeze(0), matched_node_id)
            # num_stage_one  += 1
            
            # _, sorted_logic_indices = logics.sort()
            # sorted_dist, sorted_dist_indices = dist.sort() # [nodes_per_roi,]
            
            x1y1  = pred_goals[:,:2] - 0.25 # self.config["cls_th"]
            x2y2  = pred_goals[:,:2] + 0.25 # self.config["cls_th"]
            boxes = torch.cat([x1y1, x2y2], -1)
            idxs  = torchvision.ops.nms(boxes, logics, iou_threshold=0.5)
            
            if True and len(idxs) < num_mods:
                idxs = torch.arange(len(boxes))
            
            dist = pred_goals[idxs,:2] - gt_preds[agt_i, -1].reshape(1,-1) # [nms, 2] - [1,2]
            dist = torch.sqrt(dist**2).sum(-1) # [nms,]
            min_idx = torch.argmin(dist).reshape(1)
            
            scores = logics[idxs]
            scores = F.log_softmax(scores, dim=-1)
            stage_one_loss += F.nll_loss(scores.unsqueeze(0), min_idx)
            num_stage_one  += 1
            
            _, sort_indices = dist.sort() # [nms,]
            sort_indices = idxs[sort_indices]
            
            top_k_ind = sort_indices[:num_mods]
            logics_list.append(logics[top_k_ind])
            preds_list.append(pred_goals[top_k_ind])

        loss_out["stage_one_loss"] = 0 # 0 if num_stage_one == 0 else stage_one_loss / (num_stage_one + 1e-10)
        
        
        ## here preds are not good
        preds  = torch.cat([x.unsqueeze(0) for x in preds_list], 0)    # [NAs, 6, 4] goal x/y/dx/dy
        logics = torch.cat([x.reshape(1, -1) for x in logics_list], 0) # [NAs, 6]
        # print(" preds: {}".format(preds))

        ## a1 = (2 * x_t * dx_0 + 2 * x_0 * dx_0) / (2 + dx_0 - dx_t); a0 = x_t - x_0 - a1;  a2 = x_0 
        preds     = preds.view(-1, 6, 4)     # [NAs, 6, 4], goal x4, [x_t y_t dx_t dy_t]
        agt_poses = agt_poses.view(-1, 1, 4) # [NAs, 1, 4], pose x4, [x_0 y_0 dx_0 dy_0]
        ## compute coeff, a0,a1,a2: [NAs, 6]
        a1 = (2 * preds[:,:, 0] * agt_poses[:,:, 2] + 2 * agt_poses[:,:, 0] * agt_poses[:,:, 2]) / (2 + agt_poses[:,:, 2] - preds[:,:, 2])
        a0 = preds[:,:, 0] - agt_poses[:,:, 0] - a1
        a2 = agt_poses[:,:, 0].repeat(1, 6)
        b1 = (2 * preds[:,:, 1] * agt_poses[:,:, 3] + 2 * agt_poses[:,:, 1] * agt_poses[:,:, 3]) / (2 + agt_poses[:,:, 3] - preds[:,:, 3])
        b0 = preds[:,:, 1] - agt_poses[:,:, 1] - b1
        b2 = agt_poses[:,:, 1].repeat(1, 6)

        s_samples  = (1.0/29) * torch.arange(0, 30) # [30], sample=0~1, 30 step.
        s_samples  = gpu(s_samples)
        a0, a1, a2 = a0.unsqueeze(2), a1.unsqueeze(2), a2.unsqueeze(2) # [NAs, 6, 1]
        b0, b1, b2 = b0.unsqueeze(2), b1.unsqueeze(2), b2.unsqueeze(2)
        x_samples  = a0 * s_samples**2 + a1 * s_samples + a2 # [NAs, 6, 30]
        y_samples  = b0 * s_samples**2 + b1 * s_samples + b2
        pred_trajs = torch.cat([x_samples.unsqueeze(3), y_samples.unsqueeze(3)], -1) # [NAs, 6, 30, 2]


        #### logic
        # loss_out = dict()
        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device) / float(num_preds) # [NAs, 30]
        max_last, last_idcs = last.max(1) # [NAs,] last_idcs means last valid gt_step
        mask = max_last > 1.0 # [NAs,] false means this agent has no gt

        # VAs : num of valid agents (sum along batch)
        masked_logics    = logics[mask] # [VAs, 6]
        masked_preds     = pred_trajs[mask] # [VAs, 6, 30, 2]
        masked_gt_preds  = gt_preds[mask]  # [VAs, 30, 2]
        masked_has_preds = has_preds[mask] # [VAs, 30]
        last_idcs        = last_idcs[mask]      # [VAs,]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist_per_mode = [] # 6x[VAs,]
        for imode in range(num_mods):
            dist_per_mode.append(
                torch.sqrt(
                    ((
                        masked_preds[row_idcs, imode, last_idcs]
                        - masked_gt_preds[row_idcs, last_idcs] # [VAs, 2]
                    )**2).sum(-1)
                ))
        # compute dist between each mode(pred goal point) and gt(last valid point) 
        dist = torch.cat([x.unsqueeze(1) for x in dist_per_mode], 1) # [VAs, 6]
        # get mode id with smallest dist to gt
        min_dist, min_idcs = dist.min(-1) # [VAs,]
        '''
        # get each agent's logic diff between modes and best mode, [VAs, 1] - [VAs, 6] = [VAs, 6]
        mgn = masked_logics[row_idcs, min_idcs].unsqueeze(1) - masked_logics
        # get agent whose best_mode_dist smaller than cls_th
        mask_close = (min_dist < self.config["cls_th"]).view(-1, 1) # [VAs, 1] cls_th = 2.0
        # for each agent, ignore the modes which is similar to best_mode
        mask_significant = dist - min_dist.view(-1, 1) > self.config["cls_ignore"] # [VAs, 6]
        mgn = mgn[mask_close * mask_significant]

        mask = mgn < self.config["mgn"] # 0.2
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] = coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"]  = mask.sum().item()
        '''
        
        #### BCE loss
        # final_logics = masked_logics[row_idcs, min_idcs]
        # self.sigmoid = nn.Sigmoid()
        # masked_logics = self.sigmoid(masked_logics) # [VAs, 6]
        logic_gt = torch.zeros_like(masked_logics) # [VAs, 6]
        logic_gt[row_idcs, min_idcs] = 1
        
        loss_out["cls_loss"] = F.binary_cross_entropy_with_logits(
            masked_logics,
            logic_gt.to(torch.float32),
            reduction="sum",
        )
        # loss_out["cls_loss"] = self.bce_loss(masked_logics, logic_gt)
        loss_out["num_cls"] = len(masked_logics)
        

        #### regression
        temp = [0]
        temp.extend(subgraph["num_atgs_per_batch"][:-1])
        loss_out["traj_to_eval"] = [pred_trajs[i] for i in temp] # [6, 30, 2]
        
        # masked_preds     = pred_trajs[mask] # [VAs, 6, 30, 2]
        # masked_gt_preds  = gt_preds[mask]  # [VAs, 30, 2]
        
        gt_preds   = masked_gt_preds[masked_has_preds]
        pred_trajs = masked_preds[row_idcs, min_idcs] # [VAs, 30, 2]
        pred_trajs = pred_trajs[masked_has_preds]
        
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] = coef * self.reg_loss(pred_trajs, gt_preds)
        loss_out["num_reg"]  = masked_has_preds.sum().item()
        
        if False:
            gt_preds   = gt_preds.unsqueeze(1).repeat(1, num_mods, 1, 1) # [NAs, 6, 30, 2]
            has_preds  = has_preds.unsqueeze(1).repeat(1, num_mods, 1)   # [NAs, 6, 30]
            gt_preds   = gt_preds[has_preds]
            pred_trajs = pred_trajs[has_preds]

            coef = self.config["reg_coef"]
            loss_out["reg_loss"] = coef * self.reg_loss(pred_trajs, gt_preds)
            loss_out["num_reg"]  = has_preds.sum().item()

        return loss_out


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = RoiLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(
            data, out, 
            gpu(data["gt_preds"]), 
            gpu(data["has_preds"]))

        loss_out["loss"] = loss_out["cls_loss"] / (
            loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10
        ) + loss_out["stage_one_loss"]

        return loss_out


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out, data, loss_out):
        # preds      = out["roi_feat"]
        # subgraph   = out['graphRoI']
        # loss_out["traj_to_eval"]
        post_out = dict()
        # post_out["preds"]     = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["preds"]     = [x.detach().cpu().numpy().reshape(1,6,30,2) for x in loss_out["traj_to_eval"]] # [1,6,30,2]
        
        # gt_preds = data["gt_preds"]
        # rots, origs = data["rot"], data["orig"]
        # for batch_i in range(len(gt_preds)):
        #     orig, rot = origs[batch_i], rots[batch_i]
        #     gt_preds[batch_i] = torch.matmul( (gt_preds[batch_i] - orig.view(1, 1, -1)), rot.transpose(1, 0) )
        
        
        # post_out["gt_preds"]  = [x[0:1].numpy() for x in gt_preds]  # [1,30,2]
        post_out["gt_preds"]  = [x[0:1].numpy() for x in data["gt_preds"]]  # [1,30,2]
        post_out["has_preds"] = [x[0:1].numpy() for x in data["has_preds"]] # [1,30]
        return post_out

    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]]=None) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if key == "traj_to_eval":
                metrics[key] = loss_out[key]
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg
        print("reg_loss: {:.4f}, num_reg: {}".format(metrics["reg_loss"], metrics["num_reg"]))
        stg1_cls = metrics["stage_one_loss"]
        # print("traj: {}".format(metrics["traj_to_eval"]))

        preds = np.concatenate(metrics["preds"], 0) # [N,6,30,2]
        gt_preds = np.concatenate(metrics["gt_preds"], 0) # [N,30,2]
        has_preds = np.concatenate(metrics["has_preds"], 0) # [N,30]
        ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        print(
            "loss %2.4f %2.4f %2.4f, %2.4f - ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
            % (loss, cls, reg, stg1_cls,
               ade1, fde1, ade, fde)
        )
        print()


def pred_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds    = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :, -1].argmin(1) # idx of mode [N,]
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs] # [N,1]
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade1, fde1, ade, fde, min_idcs


def get_model():
    net = Net(config)
    net = net.cuda()

    loss = Loss(config).cuda()
    post_process = PostProcess(config).cuda()

    params = net.parameters()
    opt = Optimizer(params, config)


    return config, ArgoDataset, collate_fn, net, loss, post_process, opt
