
import enum
import pstats
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
from torch import linalg as linalg

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
config["opt"] = "adamw"
config["num_epochs"] = 36
config["lr"] = [1e-3, 1e-4] #  [0.0005, 0.00005]
config["lr_epochs"] = [32]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])
config["weight_decay"] = 0.01

if "save_dir" not in config:
    config["save_dir"] = os.path.join(root_path, "results", model_name)
if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "results", config["save_dir"])

config["batch_size"] = 10
config["val_batch_size"] = 10
config["workers"] = 0
config["val_workers"] = config["workers"]

"""Dataset"""
# Raw Dataset
config["train_split"] = os.path.join(root_path, "dataset/train/data")
config["val_split"] = os.path.join(root_path, "dataset/val/data")
config["test_split"] = os.path.join(root_path, "dataset/test_obs/data")
# Preprocessed Dataset
config["preprocess"] = True # whether use preprocess or not
config["preprocess_train"] = os.path.join(root_path, "dataset","preprocess", "train_crs_dist6_angle90.p")
config["preprocess_val"]   = os.path.join(root_path,"dataset", "preprocess", "val_crs_dist6_angle90.p")
config['preprocess_test']  = os.path.join(root_path, "dataset",'preprocess', 'test_test.p')
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
        # self.pred_head = PredHead(config)
        self.decode = Decode(config)
        # self.refine_head = RefineHead(config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:        
        # construct map features
        graph = graph_gather(to_long(gpu(data["graph"])))
        # construct laneRoI features
        graphRoI = subgraph_gather(to_long(gpu(data["subgraphs"])))

        ##
        roi_feat = self.input(graphRoI)
        ##
        roi_feat = self.roi_net1(roi_feat, graphRoI)
        roi_feat = self.interactor(graph, graphRoI, roi_feat)
        roi_feat = self.roi_net2(roi_feat, graphRoI)

        # out['pred_goals'] = self.pred_head(roi_feat)
        # out['pred_refinements'] = self.refine_head(roi_feat)

        pred_logics, pred_goals, pred_trajs = self.decode(roi_feat, graphRoI, data)
        # out['graphRoI'] = graphRoI
        out = dict()
        out['pred_logics'] = pred_logics
        out['pred_goals'] = pred_goals
        out['pred_trajs'] = pred_trajs
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
    agt_vels = []
    idx_roi = 0
    interest_roi = []
    for batch_i in range(batch_size):
        # subgraphs = subgraphs_in_batch[batch_i]
        num_atgs  = len(subgraphs_in_batch[batch_i])
        num_atgs_per_batch.append(num_atgs)
        num_nodes_this_batch = 0
        for atg_i in range(num_atgs):
            counts_per_agt.append(count)
            num_nodes = len( subgraphs_in_batch[batch_i][atg_i]['feats'] )
            idcs = gpu( torch.arange(count, count + num_nodes) )
            agt_vels.append(subgraphs_in_batch[batch_i][atg_i]['agent_vel'])
            node_idcs.append(idcs)
            count += num_nodes
            num_nodes_this_batch += num_nodes
            if atg_i == 0:
                interest_roi.append(idx_roi)
            idx_roi += 1
        spans.append([start, start + num_nodes_this_batch])
        start += num_nodes_this_batch
  
    graph["num_nodes"]   = count
    graph["node_idcs"]   = torch.cat(node_idcs, 0)
    graph["counts"]      = counts_per_agt # offset per laneRoi
    graph["batch_spans"] = spans  # batch span
    graph["num_atgs_per_batch"] = num_atgs_per_batch

    temp = copy.deepcopy(counts_per_agt)
    temp.append(count)
    graph["roi_spans"] = [[temp[i], temp[i+1]] for i in range(len(temp)-1)]
    graph["interest_roi"] = torch.tensor(interest_roi, dtype=torch.long)

    idx_roi = 0
    feats, agt_feat, rel_a2m_us, rel_a2m_vs = [], [], [], []
    for batch_i in range(batch_size):
        temp, agt_temp = [], []
        for atg_i in range(num_atgs_per_batch[batch_i]):
            temp.append( subgraphs_in_batch[batch_i][atg_i]["feats"] ) # [nodes, map_dim]
            agt_temp.append( subgraphs_in_batch[batch_i][atg_i]["agent_feat"].view(1, -1) ) # [1, agt_dim]

            us = subgraphs_in_batch[batch_i][atg_i]['a2m']['u'] + idx_roi
            vs = subgraphs_in_batch[batch_i][atg_i]['a2m']['v'] + counts_per_agt[idx_roi]
            rel_a2m_us.append(us.long())
            rel_a2m_vs.append(vs.long())
            idx_roi += 1

        # merge laneRoi with a batch
        assert len(temp) > 0, "batch {} have empty subgraphs".format(batch_i)
        
        temp = torch.cat(temp, 0)
        agt_temp = torch.cat(agt_temp, 0)
        feats.append(temp)
        agt_feat.append(agt_temp)
    graph["feats"] = feats # list of tensor, size=batch_size
    graph["agent_feat"] = agt_feat # list of tensor, size=batch_size
    graph["ctrs"]  = [feats[i][:, :2] for i in range(batch_size)]
    graph["dirs"]  = [feats[i][:, 2:4] for i in range(batch_size)]
    graph["pose"]  = [feats[i][:, :4] for i in range(batch_size)]
    graph["agent_vel"] = agt_vels
    graph["a2m"] = {"u": torch.cat(rel_a2m_us, 0), "v": torch.cat(rel_a2m_vs, 0)}

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
        
#         self.map_input = nn.Sequential(
#             # nn.Linear(8, map_dim),
#             # nn.ReLU(inplace=True),
#             Linear(map_dim, map_dim, norm=norm, ng=ng),
#             # nn.Linear(map_dim, map_dim, bias=False),
#         )
#         self.agt_input = nn.Sequential(
#             # nn.Linear(80, map_dim),
#             # nn.ReLU(inplace=True),
#             Linear(map_dim, map_dim, norm=norm, ng=ng),
#             # nn.Linear(map_dim, map_dim, bias=False),
#         )
        # self.map_input = Linear(8, map_dim, norm=norm, ng=ng)
        # self.agt_input = Linear(80, map_dim, norm=norm, ng=ng)
        
        self.map_fc = nn.Linear(8, map_dim, bias=False)
        self.agt_fc = nn.Linear(80, map_dim, bias=False)
        # self.fc = Linear(2*map_dim, map_dim, norm=norm, ng=ng, act=False)
        # self.fc = Linear(map_dim, map_dim, norm=norm, ng=ng, act=False)
        self.bn = nn.GroupNorm(gcd(ng, map_dim), map_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, graph):
        map_feats = torch.cat(graph["feats"], 0)      # [nodes, 8]
        agt_feats = torch.cat(graph["agent_feat"], 0) # [agts, 80]

        # map_feats = self.map_input(map_feats) # [nodes, 128], fc+relu+(fc+gn+relu)
        # agt_feats = self.agt_input(agt_feats) # [agts, 128]
        
        # res = map_feats

        # tmp_feats = torch.zeros(map_feats.shape, dtype=map_feats.dtype, device=map_feats.device)
        map_feats = self.map_fc(map_feats)
        map_feats.index_add_(
            0, graph["a2m"]["v"],
            self.agt_fc( agt_feats[ graph["a2m"]["u"] ] ), # fc, many repeated act, we can move it outside
        )
        map_feats = self.bn(map_feats)
        map_feats = self.relu(map_feats)
        
        '''
        map_feats.index_add_(
            0, 
            graph["a2m"]["v"],
            agt_feats[ graph["a2m"]["u"] ])
        map_feats = self.bn(map_feats)
        map_feats = self.relu(map_feats)
        '''
        
        '''
        map_feats.index_add_(
            0,
            graph["a2m"]["v"],
            self.agt_fc( agt_feats[ graph["a2m"]["u"] ] ), # fc
        )
        map_feats = self.bn(map_feats)
        map_feats = self.relu(map_feats)
        '''
        
        # map_feats = torch.cat([map_feats, tmp_feats], -1)
        # map_feats = self.fc(map_feats)
        # map_feats += res
        # map_feats = self.relu(map_feats)

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
    def forward(self, context_feat, context_graph, target_feat, target_graph, dist_th=6.0, g2r=False):
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
        
        identity = target_feat
        # do not add this step will make input be changed
        # otherwise bp update target_feat that is also input of another network
        target_feat = self.input(target_feat) # fc
        
        target_feat.index_add_(0, wi, ctx) # add context feature to target feature
        target_feat = self.norm(target_feat)
        target_feat = self.relu(target_feat)

        target_feat = self.mlp(target_feat) # M_b: fc + relu + fc + norm
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
        ctrs = torch.cat(graph["ctrs"], 0)
        graph_input =  self.input(ctrs)           # [g_nodes, 2] -> [g_nodes, 128]
        graph_input += self.seg(graph["feats"])  # [g_nodes, 2] -> [g_nodes, 128]
        graph_input =  self.relu(graph_input)
        
        # graph_input = torch.zeros(( len(graph['feats']), 128 ), dtype=torch.float, device=roi_feat.device)

        graph_feat = self.roi2graph(roi_feat, subgraph, graph_input, graph) # lane pooling: roi feat -> graph feat
        graph_feat = self.global_graph_net(graph_feat, graph) # map net: graph feat -> graph feat
        roi_feat   = self.graph2roi(graph_feat, graph, roi_feat, subgraph) # lane pooling: graph feat -> roi feat
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
            Linear(n_actor, n_actor, norm=norm, ng=ng),
            nn.Linear(n_actor, 5),
        )
    
    def forward(self, roi_feat):
        out = self.pred(roi_feat) # [roi_nodes, 128] => [roi_nodes, 5]
        return out

class RefineHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        norm = "GN"
        ng = 1
        self.k = config["num_mods"] # 6
        n_actor = config["n_actor"] # 128
        out_dim = 6 * 30 * 2
        self.pred = nn.Sequential(
            Linear(n_actor, n_actor, norm=norm, ng=ng),
            nn.Linear(n_actor, out_dim),
        )
    
    def forward(self, roi_feat):
        out = self.pred(roi_feat) # [roi_nodes, 128] => [roi_nodes, 6 * 30 * 2]
        return out.reshape(-1, 6, 30, 2)


def compute_min_distance(point_set, point):
    return torch.min( torch.sqrt(((point_set - point.view(1, -1)) ** 2).sum(-1)) )

# xys: [N, 2], logits: [N]
def nms_select(xys, logits, threshold=2.0, min_len=6):
    pass
    _, idcs = logits.sort(descending=True)
    xys = xys[idcs]
    selected_xys  = torch.zeros((0,2), dtype=xys.dtype, device=xys.device)
    selected_idcs = torch.zeros((0,1), dtype=idcs.dtype, device=idcs.device)
    for i, xy in zip(idcs, xys):
        if len(selected_xys) > 0 and compute_min_distance(selected_xys, xy) < threshold:
            continue
        selected_xys  = torch.vstack([selected_xys, xy])
        selected_idcs = torch.vstack([selected_idcs, i])
    
    remain_num = min_len - len(selected_idcs)
    if remain_num > 0:
        for i in idcs:
            if remain_num > 0 and i not in selected_idcs:
                selected_idcs = torch.vstack([selected_idcs, i])
                remain_num -= 1
                if remain_num == 0:
                    break

    return selected_idcs.reshape(-1)

def compute_coefficent(agt_ctrs, agt_dirs, pred_ctrs, pred_dirs):
    # input is norm tagent vector 
    agt_ctrs = agt_ctrs.view(-1, 1, 2) # [NAs, 1, 2]
    agt_dirs = agt_dirs.view(-1, 1, 2) # [NAs, 1, 2]
    # pred_ctrs, pred_dirs # shape=[NAs, 6, 2]
    a1 = (2 * pred_ctrs[:,:,0] * agt_dirs[:,:,0] + 2 * agt_ctrs[:,:,0] * agt_dirs[:,:,0]) / (2 + agt_dirs[:,:,0] - pred_dirs[:,:,0])
    a0 = pred_ctrs[:,:,0] - agt_ctrs[:,:,0] - a1
    a2 = agt_ctrs[:,:,0].repeat(1, 6)
    b1 = (2 * pred_ctrs[:,:,1] * agt_dirs[:,:,1] + 2 * agt_ctrs[:,:,1] * agt_dirs[:,:,1]) / (2 + agt_dirs[:,:,1] - pred_dirs[:,:,1])
    b0 = pred_ctrs[:,:,1] - agt_ctrs[:,:,1] - b1
    b2 = agt_ctrs[:,:,1].repeat(1, 6)
    a0, a1, a2 = a0.unsqueeze(2), a1.unsqueeze(2), a2.unsqueeze(2) # [NAs, 6, 1]
    b0, b1, b2 = b0.unsqueeze(2), b1.unsqueeze(2), b2.unsqueeze(2)
    return a0, a1, a2, b0, b1, b2

# input:
#   a/b: [NAs, 6, 1] 
# output: [NAs, 6, 30, 2]
def sample_trajectory(s_samples, a0, a1, a2, b0, b1, b2):
    x_samples  = a0 * s_samples**2 + a1 * s_samples + a2 # [NAs, 6, 30]
    y_samples  = b0 * s_samples**2 + b1 * s_samples + b2
    pred_trajs = torch.cat([x_samples.unsqueeze(3), y_samples.unsqueeze(3)], -1) # [NAs, 6, 30, 2]
    return pred_trajs

def sample_d1_trajectory(s_samples, a0, a1, a2, b0, b1, b2):
    x_samples = 2 * a0 * s_samples + a1
    y_samples = 2 * b0 * s_samples + b1
    return torch.cat([x_samples.unsqueeze(3), y_samples.unsqueeze(3)], -1) # [NAs, 6, 30, 2]


class Decode(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        norm = "GN"
        ng = 1
        n_actor = config["n_actor"] # 128

        self.pred = nn.Sequential(
            Linear(n_actor, n_actor, norm=norm, ng=ng),
            nn.Linear(n_actor, 5),
        )
        self.agt_layer1 = nn.Sequential(
            nn.Linear(2, n_actor),
            nn.ReLU(inplace=True),
            Linear(n_actor, n_actor, norm=norm, ng=ng, act=False)
        )
        self.agt_layer2 = nn.Sequential(
            nn.Linear(2, n_actor),
            nn.ReLU(inplace=True),
            Linear(n_actor, n_actor, norm=norm, ng=ng, act=False)
        )
        self.relu = nn.ReLU(inplace=True)
        self.lane_pool = LanePooling(n_actor, n_actor)
        self.refinement = nn.Sequential(
            Linear(n_actor, n_actor, norm=norm, ng=ng),
            nn.Linear(n_actor, 30 * 2),
        ) # 6*30*2
    
    # input:
    #   roi_feat: [roi_nodes, 128]
    #   subgraph: laneRoI of mini-batch
    #   data: input
    def forward(self, roi_feat, subgraph, data):
        # only predict interest agent
        interest_roi_idcs = subgraph["interest_roi"]
        interest_roi_feats = [roi_feat[it[0]:it[1]] for i, it in enumerate(subgraph["roi_spans"]) 
            if i in interest_roi_idcs]
        
        iagt_roi_spans = [0]
        count = 0
        for it in interest_roi_feats:
            count += len(it)
            iagt_roi_spans.append(count)
        interest_roi_feats = torch.cat(interest_roi_feats, dim=0)
        pred_goals = self.pred( interest_roi_feats )

        #### Decode
        #### input: data, out
        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"] # 6, 30
        device     = pred_goals.device
        valid_agent_ids = to_long(gpu(data['valid_agent_ids']))

        # map anchor
        anchor_ctrs = torch.cat(subgraph["ctrs"], 0)
        anchor_dirs = torch.cat(subgraph["dirs"], 0)
        interest_roi_ctrs, interest_roi_dirs = [], []

        pred_goals_per_agt, pred_thetas_per_agt, pred_logics_per_agt  = [], [], []
        top_k_idcs_per_agt = []
        # for each laneRoI (number = NAs) interest_roi_spans
        # for agt_i, span in enumerate(subgraph["roi_spans"]):
        for i, roi_id in enumerate(interest_roi_idcs):
            span = subgraph["roi_spans"][roi_id.item()]
            ### get anchor
            anc_ctrs = anchor_ctrs[span[0] : span[1]] # [nodes_per_roi, 2]
            anc_dirs = anchor_dirs[span[0] : span[1]] # [nodes_per_roi, 2]
            anc_theta = torch.atan2(anc_dirs[:, 1], anc_dirs[:, 0]) 
            ### generate prediction with anchor
            # pred    = pred_goals[span[0] : span[1]]   # [nodes_per_roi, 5]
            pred = pred_goals[iagt_roi_spans[i] : iagt_roi_spans[i+1]]
            logics  = pred[:, 0]  # [nodes_per_roi,]
            pred_delta_xy    = pred[:, 1:3] # [nodes_per_roi, 2]
            pred_delta_theta = torch.atan(pred[:, 3] / pred[:, 4]) # [nodes_per_roi,]
            
            pred_xy = anc_ctrs + pred_delta_xy # [nodes_per_roi, 2]
            pred_theta = anc_theta + pred_delta_theta # [nodes_per_roi,]

            nms_ids = nms_select(pred_xy, logics)
            top_k_idcs = nms_ids[:num_mods]

            pred_goals_per_agt.append(pred_xy[top_k_idcs]) # append [6, 2]
            pred_thetas_per_agt.append(pred_theta[top_k_idcs]) # append [6,]
            pred_logics_per_agt.append(logics[top_k_idcs]) # append [6,]
            interest_roi_ctrs.append(anc_ctrs)
            interest_roi_dirs.append(anc_dirs)
            top_k_idcs_per_agt.append(top_k_idcs)
        
        # NAs: number of interest agents
        pred_ctrs   = torch.cat([x.unsqueeze(0) for x in pred_goals_per_agt], 0) # [NAs, 6, 2] predicted goal x/y
        pred_thetas = torch.cat([x.unsqueeze(0) for x in pred_thetas_per_agt], 0) # [NAs, 6]
        pred_logics = torch.cat([x.unsqueeze(0) for x in pred_logics_per_agt], 0) # [NAs, 6]
        pred_norm_dirs = torch.cat([torch.cos(pred_thetas).view(-1, num_mods, 1), 
                                    torch.sin(pred_thetas).view(-1, num_mods, 1)], -1) # [NAs, 6, 2]

        list_agt_ctrs  = [ x[ids][0].view(-1,2)    for ids, x in zip( valid_agent_ids, gpu(data['ctrs']) )]
        list_agt_dirs  = [ x[ids][0].view(-1,20,3)[:,:,:2] for ids, x in zip( valid_agent_ids, gpu(data['feats']) )]
        list_agt_trajs = [ x[ids][0].view(-1,20,3)[:,:,:2] for ids, x in zip( valid_agent_ids, gpu(data['obs_trajs']) )]
        
        agt_ctrs  = torch.cat(list_agt_ctrs, 0)  # [NAs, 2]
        agt_dirs  = torch.cat(list_agt_dirs, 0)  # [NAs, 20, 2]
        agt_trajs = torch.cat(list_agt_trajs, 0) # [NAs, 20, 2]
        agt_vels = subgraph["agent_vel"] # list
        agt_vels = torch.tensor(agt_vels, dtype=agt_ctrs.dtype, device=device)[interest_roi_idcs] # [NAs,] to do: get IAs not all

        agt_final_dir  = agt_dirs[:, -1, :2] # [NAs, 2]
        norm_dist = linalg.norm(agt_final_dir, dim=1) # [NAs,]
        agt_norm_dirs = agt_final_dir / norm_dist.view(-1, 1) # [NAs, 2] / [NAs, 1]
        agt_norm_dirs[norm_dist < 1e-6] = 0.
        a0, a1, a2, b0, b1, b2 = compute_coefficent(agt_ctrs, agt_norm_dirs, pred_ctrs, pred_norm_dirs) # a/b: [NAs, 6, 1]

        s_samples  = (1.0/30) * torch.arange(0, 31).float().to(device)
        pred_trajs = sample_trajectory(s_samples, a0, a1, a2, b0, b1, b2) # [NAs, 6, 30, 2]

        pred_dists = pred_trajs[:,:,1:] - pred_trajs[:,:,:-1]
        pred_dists = ( torch.sqrt((pred_dists**2).sum(-1)) ).sum(-1) # [NAs, 6]
        pred_accs = 2 * (pred_dists - agt_vels.view(-1, 1) * 3.0) / 9.0 # [NAs, 6]

        t_samples = 0.1 * torch.arange(0, 31).float().to(device) # shape=[31], val=[0, 0,1 ... 3.0]
        v_samples = agt_vels.view(-1, 1, 1) + pred_accs.view(-1, 6, 1) * t_samples # [NAs,6,31]
        v_samples[v_samples <= 0.] = 0.
        s_samples = (v_samples[:,:,0].unsqueeze(2) + v_samples[:,:,1:]) * t_samples[1:] / 2 # [NAs,6,30]
        max_vals, max_idcs = s_samples.max(2) # [NAs, 6]
        s_samples_ = s_samples / max_vals.unsqueeze(2) # [NAs,6,30]
        s_samples_[s_samples_ == 0.0] = 1.0
        pred_trajs = sample_trajectory(s_samples_, a0, a1, a2, b0, b1, b2) # [NAs, 6, 30, 2]



        # interest_roi_feats: shape=[nodes_of_IAs, 128]
        # iagt_roi_spans[:-1]: count of nodes per IAgent
        # agt_trajs[:,:,:2], agt_dirs[:,:,:2]
        # create two graphs:
        graph_map = dict()
        graph_map['ctrs'] = interest_roi_ctrs
        graph_map['pose'] = [torch.cat([a, b], dim=-1) 
            for a, b in zip(interest_roi_ctrs, interest_roi_dirs)] # list_size=mini_batch
        graph_agt_movement = dict()
        graph_agt_movement['ctrs'] = list_agt_trajs
        graph_agt_movement['pose'] = [torch.cat([a.view(-1,2), b.view(-1,2)], dim=-1) 
            for a, b in zip(list_agt_trajs, list_agt_dirs)] # list_size=mini_batch

        
        agt_feat = self.agt_layer1(agt_trajs.view(-1, 2)) # [NAs*20, 128]
        agt_feat += self.agt_layer2(agt_dirs.view(-1, 2)) # [NAs*20, 128]
        agt_feat = self.relu(agt_feat)
        interest_roi_feats = self.lane_pool(
            agt_feat, graph_agt_movement, interest_roi_feats, graph_map)

        batch_size = len(interest_roi_ctrs)
        traj_feats = []
        for i in range(batch_size):
            temp = interest_roi_feats[iagt_roi_spans[i] : iagt_roi_spans[i+1]]
            traj_feats.append( temp[ top_k_idcs_per_agt[i] ].unsqueeze(0) ) # [1, 6, 128]
        traj_feats = torch.cat(traj_feats, 0) # [NAs, 6, 128]
        bs = len(traj_feats)
        traj_delta = self.refinement(traj_feats.view(bs*6, -1)).view(bs, 6, -1) # [NAs, 6, 128] => [NAs, 6, 30 * 2]
        traj_delta = traj_delta.view(-1, 6, 30, 2) # [NAs, 6, 30, 2]

        # [NAs,6,30] + [NAs,6,30] => [NAs,6,30]
        s_samples = s_samples + traj_delta[:,:,:,0]
        max_vals, _ = s_samples.max(2) # [NAs, 6]
        s_samples_ = s_samples / max_vals.unsqueeze(2) # [NAs, 6, 30]
        s_samples_[s_samples_ == 0.0] = 1.0
        dxy_samples = sample_d1_trajectory(s_samples_, a0, a1, a2, b0, b1, b2) # [NAs, 6, 30, 2]

        rotate_ = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dxy_samples.dtype, device=dxy_samples.device)
        bs = dxy_samples.shape[0]
        # rotate_ = rotate_.reshape(1,2,2).repeat(bs*6*30, 1, 1) # [NAs*6*30, 2, 2]
        # use transpose of rorate for negative d_delta
        d_deltas = traj_delta[:,:,:,1] # [NAs, 6, 30]
        # mask = (d_deltas < 0.0).view(-1)
        # rotate_[mask,0,1] = 1
        # rotate_[mask,1,0] = -1
        # [2,2] * [NAs*6*30,2,1] => [NAs*6*30,2,1] => [NAs,6,30,2]
        norm_dxy_samples = torch.matmul(rotate_, dxy_samples.view(bs*6*30, 2, 1)).view(bs,6,30,2)
        shift_xy_samples = norm_dxy_samples * d_deltas.unsqueeze(3) # [NAs,6,30,2]

        pred_trajs = sample_trajectory(s_samples_, a0, a1, a2, b0, b1, b2) # [NAs, 6, 30, 2]
        pred_trajs = pred_trajs + shift_xy_samples

        # a = torch.rand((1, 2))
        # torch.linalg.svd(a).Vh[-1]

        return pred_logics, pred_ctrs, pred_trajs

class RoiLossForGoals(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config   = config
        self.num_mods = config["num_mods"] # 6
        self.bce_loss = nn.BCELoss()
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    # NAs : num of agents (sum along batch)
    def forward(self,
                data,
                out: Dict, # network output
                gt_preds: List[Tensor], 
                has_preds: List[Tensor]):
        
        #### Decode
        #### input: data, out
        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"] # 6, 30
        pred_goals       = out["pred_goals"] # [?, 5]
        pred_refinements = out["pred_refinements"] # [?, 6, 30, 2]
        subgraph   = out['graphRoI']
        device     = pred_goals.device
        # batch_size = len(gt_preds)
        valid_agent_ids = to_long(gpu(data['valid_agent_ids']))


        # map anchor
        anchor_ctrs = torch.cat(subgraph["ctrs"], 0)
        anchor_dirs = torch.cat(subgraph["dirs"], 0)

        pred_goals_per_agt, pred_thetas_per_agt, pred_logics_per_agt  = [], [], []
        # for each laneRoI (number = NAs) interest_roi_spans
        for agt_i, span in enumerate(subgraph["roi_spans"]):
            ### get anchor
            anc_ctrs = anchor_ctrs[span[0] : span[1]] # [nodes_per_roi, 2]
            anc_dirs = anchor_dirs[span[0] : span[1]] # [nodes_per_roi, 2]
            anc_theta = torch.atan2(anc_dirs[:, 1], anc_dirs[:, 0]) 
            ### generate prediction with anchor
            pred    = pred_goals[span[0] : span[1]]   # [nodes_per_roi, 5]
            logics  = pred[:, 0]  # [nodes_per_roi,]
            pred_delta_xy    = pred[:, 1:3] # [nodes_per_roi, 2]
            pred_delta_theta = torch.atan(pred[:, 3] / pred[:, 4]) # [nodes_per_roi,]
            
            pred_xy = anc_ctrs + pred_delta_xy # [nodes_per_roi, 2]
            pred_theta = anc_theta + pred_delta_theta # [nodes_per_roi,]

            nms_ids = nms_select(pred_xy, logics)
            top_k_idcs = nms_ids[:num_mods]

            pred_goals_per_agt.append(pred_xy[top_k_idcs]) # append [6, 2]
            pred_thetas_per_agt.append(pred_theta[top_k_idcs]) # append [6,]
            pred_logics_per_agt.append(logics[top_k_idcs]) # append [6,]
        
        pred_ctrs   = torch.cat([x.unsqueeze(0) for x in pred_goals_per_agt], 0) # [NAs, 6, 2] predicted goal x/y
        pred_thetas = torch.cat([x.unsqueeze(0) for x in pred_thetas_per_agt], 0) # [NAs, 6]
        pred_logics = torch.cat([x.unsqueeze(0) for x in pred_logics_per_agt], 0) # [NAs, 6]
        pred_norm_dirs = torch.cat([torch.cos(pred_thetas).view(-1, num_mods, 1), 
                                    torch.sin(pred_thetas).view(-1, num_mods, 1)], -1) # [NAs, 6, 2]



        agt_ctrs = [x[ids].view(-1,2)    for ids, x in zip( valid_agent_ids, gpu(data['ctrs']) )]
        agt_dirs = [x[ids].view(-1,20,3) for ids, x in zip( valid_agent_ids, gpu(data['feats']) )]
        agt_ctrs = torch.cat(agt_ctrs, 0) # [NAs, 2]
        agt_dirs = torch.cat(agt_dirs, 0) # [NAs, 20, 3]
        agt_vels = subgraph["agent_vel"] # list
        agt_vels = torch.tensor(agt_vels, dtype=agt_ctrs.dtype, device=device) # [NAs,]

        agt_final_dir  = agt_dirs[:, -1, :2] # [NAs, 2]
        norm_dist = linalg.norm(agt_final_dir, dim=1) # [NAs,]
        agt_norm_dirs = agt_final_dir / norm_dist.view(-1, 1) # [NAs, 2] / [NAs, 1]
        agt_norm_dirs[norm_dist < 1e-6] = 0.
        a0, a1, a2, b0, b1, b2 = compute_coefficent(agt_ctrs, agt_norm_dirs, pred_ctrs, pred_norm_dirs) # a/b: [NAs, 6, 1]

        s_samples  = (1.0/30) * torch.arange(0, 31).float().to(device)
        pred_trajs = sample_trajectory(s_samples, a0, a1, a2, b0, b1, b2) # [NAs, 6, 30, 2]

        pred_dists = pred_trajs[1:] - pred_trajs[:-1]
        pred_dists = ( torch.sqrt((pred_dists**2).sum(-1)) ).sum(-1) # [NAs, 6]
        pred_accs = 2 * (pred_dists - agt_vels.view(-1, 1) * 3.0) / 9.0 # [NAs, 6]

        t_samples = 0.1 * torch.arange(0, 31).float().to(device) # shape=[31], val=[0, 0,1 ... 3.0]
        v_samples = agt_vels.view(-1, 1, 1) + pred_accs.view(-1, 6, 1) * t_samples # [NAs,6,31]
        v_samples[v_samples <= 0.] = 0.
        s_samples = (v_samples[:,:,0].unsqueeze(2) + v_samples[:,:,1:]) * t_samples[1:] / 2 # [NAs,6,30]
        max_vals, max_idcs = s_samples.max(2) # [NAs, 6]
        s_samples = s_samples / max_vals.unsqueeze(2) # [NAs,6,30]
        s_samples[s_samples==0.] = 1.0
        pred_trajs = sample_trajectory(s_samples, a0, a1, a2, b0, b1, b2) # [NAs, 6, 30, 2]


        #### Comppute Loss
        #### logic
        gt_preds  = [x[ids].view(-1,30,2) for ids, x in zip( valid_agent_ids, gt_preds )]
        has_preds = [x[ids].view(-1,30)   for ids, x in zip( valid_agent_ids, has_preds )]
        gt_preds  = torch.cat(gt_preds, 0)
        has_preds = torch.cat(has_preds, 0)

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(device) / float(num_preds) # [NAs, 30]
        max_last, last_idcs = last.max(1) # [NAs,] last_idcs means last valid gt_step

        mask = subgraph["interest_roi"]




        # VAs : num of valid agents (sum along batch)
        masked_logics    = pred_logics[mask] # [VAs, 6]
        masked_preds     = pred_trajs[mask] # [VAs, 6, 30, 2]
        masked_goals     = pred_ctrs[mask] # [VAs, 6, 2]
        masked_gt_preds  = gt_preds[mask]  # [VAs, 30, 2]
        masked_has_preds = has_preds[mask] # [VAs, 30]
        last_idcs        = last_idcs[mask] # [VAs,]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist_per_mode = [] # 6x[VAs,]
        for imode in range(num_mods):
            dist_per_mode.append(
                torch.sqrt(
                    ((
                        masked_goals[row_idcs, imode] # [VAs, 2]
                        - masked_gt_preds[row_idcs, last_idcs] # [VAs, 2]
                    )**2).sum(-1)
                ))
        # compute dist between each mode(pred goal point) and gt(last valid point) 
        dist = torch.cat([x.unsqueeze(1) for x in dist_per_mode], 1) # [VAs, 6]
        # get mode id with smallest dist to gt
        _, min_idcs = dist.min(-1) # [VAs,]

        
        #### BCE loss
        logic_gt = torch.zeros_like(masked_logics) # [VAs, 6]
        logic_gt[row_idcs, min_idcs] = 1
        
        loss_out = dict()
        loss_out["cls_loss"] = F.binary_cross_entropy_with_logits(
            masked_logics,
            logic_gt.to(torch.float32),
            reduction="sum",
        )
        loss_out["num_cls"] = len(masked_logics)
        del logic_gt

        #### regression
        # temp = [0]
        # temp.extend(subgraph["num_atgs_per_batch"][:-1])
        # loss_out["traj_to_eval"] = [pred_trajs[i] for i in temp] # [6, 30, 2]


        # gt_preds   = masked_gt_preds[masked_has_preds]
        masked_has_preds = masked_has_preds[row_idcs, last_idcs] # [VAs,]
        gt_goals = masked_gt_preds[row_idcs, last_idcs] # [VAs, 2]
        gt_goals = gt_goals[ masked_has_preds ]

        pred_goals = masked_goals[row_idcs, min_idcs] # [VAs, 2]
        pred_goals = pred_goals[masked_has_preds]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] = coef * self.reg_loss(pred_goals, gt_goals)
        loss_out["num_reg"]  = masked_has_preds.sum().item()
        loss_out["stage_one_loss"] = 0
        loss_out["goals_to_eval"] = pred_goals.reshape(-1, 2) # [VAs, 2]

        return loss_out

    def forward_backup(self,
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
        gt_preds  = torch.cat(gt_preds, 0)
        has_preds = torch.cat(has_preds, 0)

        loss_out = dict()
        anchor_ctrs = torch.cat(subgraph["ctrs"], 0)
        anchor_dirs = torch.cat(subgraph["dirs"], 0)

        pred_goals_per_agt, pred_theta_per_agt, pred_logics_per_agt  = [], [], []
        # for each laneRoI (number = NAs) interest_roi_spans
        for agt_i, span in enumerate(subgraph["roi_spans"]):
            ### get anchor
            anc_ctrs = anchor_ctrs[span[0] : span[1]] # [nodes_per_roi, 2]
            anc_dirs = anchor_dirs[span[0] : span[1]] # [nodes_per_roi, 2]
            anc_theta = torch.atan2(anc_dirs[:, 1], anc_dirs[:, 0]) 
            ### generate prediction
            pred    = preds[span[0] : span[1]]   # [nodes_per_roi, 5]
            logics  = pred[:, 0]  # [nodes_per_roi,]
            pred_delta_xy    = pred[:, 1:3] # [nodes_per_roi, 2]
            pred_delta_theta = torch.atan(pred[:, 3] / pred[:, 4]) # [nodes_per_roi,]
            
            pred_xy = anc_ctrs + pred_delta_xy # [nodes_per_roi, 2]
            pred_theta = anc_theta + pred_delta_theta # [nodes_per_roi,]

            nms_ids = nms_select(pred_xy, logics)

            # top_k_idcs = sort_indices[:num_mods]
            top_k_idcs = nms_ids[:num_mods]

            pred_logics_per_agt.append(logics[top_k_idcs])
            pred_goals_per_agt.append(pred_xy[top_k_idcs])
            pred_theta_per_agt.append(pred_theta[top_k_idcs])
        
        pred_ctrs = torch.cat([x.unsqueeze(0) for x in pred_goals_per_agt], 0)  # [NAs, 6, 2] predicted goal x/y
        logics    = torch.cat([x.unsqueeze(0) for x in pred_logics_per_agt], 0) # [NAs, 6]
        
        #### logic
        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device) / float(num_preds) # [NAs, 30]
        max_last, last_idcs = last.max(1) # [NAs,] last_idcs means last valid gt_step

        mask = subgraph["interest_roi"]

        # VAs : num of valid agents (sum along batch)
        masked_logics    = logics[mask] # [VAs, 6]
        # masked_preds     = pred_trajs[mask] # [VAs, 6, 30, 2]
        masked_goals     = pred_ctrs[mask] # [VAs, 6, 2]
        masked_gt_preds  = gt_preds[mask]  # [VAs, 30, 2]
        masked_has_preds = has_preds[mask] # [VAs, 30]
        last_idcs        = last_idcs[mask] # [VAs,]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist_per_mode = [] # 6x[VAs,]
        for imode in range(num_mods):
            dist_per_mode.append(
                torch.sqrt(
                    ((
                        masked_goals[row_idcs, imode] # [VAs, 2]
                        - masked_gt_preds[row_idcs, last_idcs] # [VAs, 2]
                    )**2).sum(-1)
                ))
        # compute dist between each mode(pred goal point) and gt(last valid point) 
        dist = torch.cat([x.unsqueeze(1) for x in dist_per_mode], 1) # [VAs, 6]
        # get mode id with smallest dist to gt
        _, min_idcs = dist.min(-1) # [VAs,]

        
        #### BCE loss
        logic_gt = torch.zeros_like(masked_logics) # [VAs, 6]
        logic_gt[row_idcs, min_idcs] = 1
        
        loss_out["cls_loss"] = F.binary_cross_entropy_with_logits(
            masked_logics,
            logic_gt.to(torch.float32),
            reduction="sum",
        )
        loss_out["num_cls"] = len(masked_logics)
        del logic_gt

        #### regression
        # temp = [0]
        # temp.extend(subgraph["num_atgs_per_batch"][:-1])
        # loss_out["traj_to_eval"] = [pred_trajs[i] for i in temp] # [6, 30, 2]


        # gt_preds   = masked_gt_preds[masked_has_preds]
        masked_has_preds = masked_has_preds[row_idcs, last_idcs] # [VAs,]
        gt_goals = masked_gt_preds[row_idcs, last_idcs] # [VAs, 2]
        gt_goals = gt_goals[ masked_has_preds ]

        pred_goals = masked_goals[row_idcs, min_idcs] # [VAs, 2]
        pred_goals = pred_goals[masked_has_preds]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] = coef * self.reg_loss(pred_goals, gt_goals)
        loss_out["num_reg"]  = masked_has_preds.sum().item()
        loss_out["stage_one_loss"] = 0
        loss_out["goals_to_eval"] = pred_goals.reshape(-1, 2) # [VAs, 2]

        return loss_out

# roi loss for goal and trajectory
class RoiLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config   = config
        self.num_mods = config["num_mods"] # 6
        self.bce_loss = nn.BCELoss()
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    # NAs : num of agents (sum along batch)
    def forward(self,
                data, # input data of network (on CPU)
                out: Dict, # network output
                gt_preds: List[Tensor], 
                has_preds: List[Tensor]):
        num_preds = 30
        num_mods = 6

        pred_logics = out['pred_logics'] # [bs, 6]
        pred_goals = out['pred_goals']   # [bs, 6, 2]
        pred_trajs = out['pred_trajs']   # [bs, 6, 30, 2]
        
        valid_agent_ids = to_long(gpu(data['valid_agent_ids']))

        gt_preds  = [x[ids][0].view(-1,30,2) for ids, x in zip( valid_agent_ids, gt_preds )]
        has_preds = [x[ids][0].view(-1,30)   for ids, x in zip( valid_agent_ids, has_preds )]
        gt_preds  = torch.cat(gt_preds, 0) # [bs, 30, 2]
        has_preds = torch.cat(has_preds, 0) # [bs, 30]

        device = has_preds.device
        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(device) / float(num_preds) # [NAs, 30]
        max_last, last_idcs = last.max(1) # [NAs,] last_idcs means last valid gt_step

        # masked_logics    = pred_logics # [VAs, 6]
        # masked_preds     = pred_trajs # [VAs, 6, 30, 2]
        # masked_goals     = pred_goals # [VAs, 6, 2]

        # masked_gt_preds  = gt_preds  # [VAs, 30, 2]
        # masked_has_preds = has_preds # [VAs, 30]
        # last_idcs        = last_idcs # [VAs,]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist_per_mode = [] # 6x[VAs,]
        for imode in range(num_mods):
            dist_per_mode.append(
                torch.sqrt(
                    ((
                        pred_goals[row_idcs, imode] # [VAs, 2]
                        - gt_preds[row_idcs, last_idcs] # [VAs, 2]
                    )**2).sum(-1)
                ))
        # compute dist between each mode(pred goal point) and gt(last valid point) 
        dist = torch.cat([x.unsqueeze(1) for x in dist_per_mode], 1) # [VAs, 6]
        # get mode id with smallest dist to gt
        _, min_idcs = dist.min(-1) # [VAs,]

        #### BCE loss
        logic_gt = torch.zeros_like(pred_logics) # [VAs, 6]
        logic_gt[row_idcs, min_idcs] = 1
        
        loss_out = dict()
        loss_out["cls_loss"] = F.binary_cross_entropy_with_logits(
            pred_logics,
            logic_gt.to(torch.float32),
            reduction="sum",
        )
        loss_out["num_cls"] = len(pred_logics)
        del logic_gt

        #### Regression Loss to Goal
        has_goals = has_preds[row_idcs, last_idcs] # [VAs,]
        gt_goals = gt_preds[row_idcs, last_idcs] # [VAs, 2]
        gt_goals = gt_goals[has_goals]

        pred_goals = pred_goals[row_idcs, min_idcs] # [VAs, 2]
        loss_out["pred_goals"] = pred_goals # [VAs, 2]

        pred_goals = pred_goals[has_goals]
        coef = self.config["reg_coef"]
        loss_out["reg_goal_loss"] = coef * self.reg_loss(pred_goals, gt_goals)
        loss_out["num_reg_goal"]  = has_goals.sum().item() # = batch_size
        
        #### Regression Loss to Trajectory
        loss_out["pred_trajs"] = pred_trajs[row_idcs] # [VAs, 6, 30, 2]
        pred_trajs = pred_trajs[row_idcs, min_idcs] # [VAs, 30, 2]

        pred_trajs = pred_trajs[has_preds]
        gt_trajs = gt_preds[row_idcs] # [VAs, 30, 2]
        gt_trajs = gt_trajs[has_preds]
        loss_out["reg_traj_loss"] = coef * self.reg_loss(pred_trajs, gt_trajs)
        loss_out["num_reg_traj"]  = has_preds.sum().item() # = batch_size * 30
        
        #### others
        loss_out["stage_one_loss"] = 0
        loss_out["num_stage_one"] = 1
        # loss_out["goals_to_eval"] = pred_goals.reshape(-1, 2) # [VAs, 2]

        return loss_out

        

class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = RoiLoss(config)

    # inputs:
    #   out: net output
    #   data: net input(on cpu) should be move to GPU
    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(
            data, out, 
            gpu(data["gt_preds"]), 
            gpu(data["has_preds"]))

        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_goal_loss"] / (loss_out["num_reg_goal"] + 1e-10
        ) + loss_out["reg_traj_loss"] / (loss_out["num_reg_traj"] + 1e-10
        ) + loss_out["stage_one_loss"] / (loss_out["num_stage_one"] + 1e-10)

        return loss_out


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    # inputs
    #   out: network output
    #   data: input data of network (on CPU)
    #   loss_out: output of loss function
    def forward(self, out, data, loss_out):
        # preds      = out["roi_feat"]
        # subgraph   = out['graphRoI']
        # loss_out["traj_to_eval"]
        post_out = dict()
        # post_out["preds"]     = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        # post_out["preds"]     = [x.detach().cpu().numpy().reshape(1,6,30,2) for x in loss_out["traj_to_eval"]] # [1,6,30,2]
        # post_out["goals"] = [x.detach().cpu().numpy() for x in loss_out["pred_goals"]] # [batch, 2]
        post_out["goals"] = [ loss_out["pred_goals"].detach().cpu().numpy() ] # [batch, 2]
        post_out["trajs"] = [ loss_out["pred_trajs"].detach().cpu().numpy() ] # [batch, 6, 30, 2]

        # gt_preds = data["gt_preds"]
        # rots, origs = data["rot"], data["orig"]
        # for batch_i in range(len(gt_preds)):
        #     orig, rot = origs[batch_i], rots[batch_i]
        #     gt_preds[batch_i] = torch.matmul( (gt_preds[batch_i] - orig.view(1, 1, -1)), rot.transpose(1, 0) )
        # post_out["gt_preds"]  = [x[0:1].numpy() for x in gt_preds]  # [1,30,2]
        post_out["gt_preds"]  = [x[0:1].numpy() for x in data["gt_preds"]]  # list of tensor[1,30,2]
        post_out["has_preds"] = [x[0:1].numpy() for x in data["has_preds"]] # list of tensor[1,30]
        return post_out

    # inputs:
    #   loss_out: output of loss func, 
    #   post_out: output of forward method of PostProcess
    def append(self, 
               metrics: Dict, 
               loss_out: Dict, 
               post_out: Optional[Dict[str, List[ndarray]]]=None) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if key == "pred_goals" or key == "pred_trajs":
               # metrics[key] = loss_out[key]
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
        reg_goal = metrics["reg_goal_loss"] / (metrics["num_reg_goal"] + 1e-10)
        reg_traj = metrics["reg_traj_loss"] / (metrics["num_reg_traj"] + 1e-10)
        stg1_cls = metrics["stage_one_loss"] / (metrics["num_stage_one"] + 1e-10)
        loss = cls + reg_goal + reg_traj + stg1_cls
        # print("reg_loss: {:.4f}, num_reg: {}".format(metrics["reg_loss"], metrics["num_reg"]))
        # print("traj: {}".format(metrics["traj_to_eval"]))

        # preds = np.concatenate(metrics["preds"], 0) # [N,6,30,2]
        goals = np.concatenate(metrics["goals"], 0) # [all_batch, 2]
        trajs = np.concatenate(metrics["trajs"], 0) # [all_batch, 6, 30, 2]
        gt_preds = np.concatenate(metrics["gt_preds"], 0)   # [all_batch,30,2]
        has_preds = np.concatenate(metrics["has_preds"], 0) # [all_batch,30]
        # ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)
        # fde = pred_metrics_ade(goals, gt_preds, has_preds)
        ade1, fde1, ade, fde, min_idcs = pred_metrics(trajs, gt_preds, has_preds)

        # print(
        #     "loss %2.4f - %2.4f %2.4f %2.4f, %2.4f - fde %2.4f"
        #     % (loss, cls, reg_goal, reg_traj, stg1_cls, fde)
        # )
        print(
            "loss %2.4f - %2.4f %2.4f %2.4f, %2.4f - ade1=%2.4f fde1=%2.4f ade=%2.4f fde=%2.4f"
            % (loss, cls, reg_goal, reg_traj, stg1_cls,   ade1, fde1, ade, fde)
        )
        print()


def pred_metrics_ade(goals, gt_preds, has_preds):
    assert has_preds.all()
    goals    = np.asarray(goals, np.float32).reshape(-1, 2) # [batch, 2]
    gt_preds = np.asarray(gt_preds, np.float32) # [batch,30,2]

    """batch_size x num_mods x num_preds"""
    # err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    # ade1 = err[:, 0].mean()
    # fde1 = err[:, 0, -1].mean()

    # min_idcs = err[:, :, -1].argmin(1) # idx of mode [N,]
    # row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    # err = err[row_idcs, min_idcs] # [N,1]
    # ade = err.mean()
    # fde = err[:, -1].mean()
    fde = np.sqrt(((goals - gt_preds[:,-1]) ** 2).sum(-1)) # [batch]
    fde = fde.mean()
    return fde

def pred_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds    = np.asarray(preds, np.float32)    # [N, 6, 30, 2]
    gt_preds = np.asarray(gt_preds, np.float32) # [N, 30, 2]

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3)) # [N, 6, 30]

    # why we choose 0 here is that the first mode is predicted as most likely best (with highest logit)
    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :, -1].argmin(1) # find idx of best mode [N,]
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs] # error of best mode per agent[N,1,30]
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

def get_model_for_torch_dist():
    net = Net(config)
    net = net.cuda()

    loss = Loss(config).cuda()
    post_process = PostProcess(config).cuda()

    # params = net.parameters()
    # opt = Optimizer(params, config)


    return config, ArgoDataset, collate_fn, net, loss, post_process, # opt
