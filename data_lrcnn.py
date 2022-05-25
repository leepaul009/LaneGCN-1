
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import os
import copy
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from skimage.transform import rotate


class ArgoDataset(Dataset):
    def __init__(self, split, config, train=True):
        self.config = config
        self.train = train
        
        if 'preprocess' in config and config['preprocess']: # 当已经预处理过数据后，训练阶段进入：
            if train:
                self.split = np.load(self.config['preprocess_train'], allow_pickle=True)
                # self.split = self.split[:2500]
            else:
                self.split = np.load(self.config['preprocess_val'], allow_pickle=True)
        else: # 预处理数据：
            self.avl = ArgoverseForecastingLoader(split)
            self.am = ArgoverseMap()

        if 'raster' in config and config['raster']: # pass both preprocess and training
            #TODO: DELETE
            self.map_query = MapQuery(config['map_scale'])
            
    def __getitem__(self, idx):
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.split[idx]

            if self.train and self.config['rot_aug']:
                new_data = dict()
                for key in ['city', 'orig', 'gt_preds', 'has_preds']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])

                dt = np.random.rand() * self.config['rot_size']#np.pi * 2.0
                theta = data['theta'] + dt
                new_data['theta'] = theta
                new_data['rot'] = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)

                rot = np.asarray([
                    [np.cos(-dt), -np.sin(-dt)],
                    [np.sin(-dt), np.cos(-dt)]], np.float32)
                new_data['feats'] = data['feats'].copy()
                new_data['feats'][:, :, :2] = np.matmul(new_data['feats'][:, :, :2], rot)
                new_data['ctrs'] = np.matmul(data['ctrs'], rot)

                graph = dict()
                for key in ['num_nodes', 'turn', 'control', 'intersect', 'pre', 'suc', 'lane_idcs', 'left_pairs', 'right_pairs', 'left', 'right']:
                    graph[key] = ref_copy(data['graph'][key])
                graph['ctrs'] = np.matmul(data['graph']['ctrs'], rot)
                graph['feats'] = np.matmul(data['graph']['feats'], rot)
                new_data['graph'] = graph
                data = new_data
            else:
                new_data = dict()
                for key in ['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats','obs_trajs', 'ctrs', 'graph']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])
                        # new_data[key] = data[key]
                data = generate_lane_roi(new_data)
           
            if 'raster' in self.config and self.config['raster']:
                data.pop('graph')
                x_min, x_max, y_min, y_max = self.config['pred_range']
                cx, cy = data['orig']
                
                region = [cx + x_min, cx + x_max, cy + y_min, cy + y_max]
                raster = self.map_query.query(region, data['theta'], data['city'])

                data['raster'] = raster
            return data
        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        data['idx'] = idx

        if 'raster' in self.config and self.config['raster']:
            print("do raster.....................")
            x_min, x_max, y_min, y_max = self.config['pred_range']
            cx, cy = data['orig']

            region = [cx + x_min, cx + x_max, cy + y_min, cy + y_max]
            raster = self.map_query.query(region, data['theta'], data['city'])

            data['raster'] = raster
            return data

        #print("get_lane_graph")
        data['graph'] = self.get_lane_graph(data)
        return data
    
    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return len(self.avl)

    def read_argo_data(self, idx):
        city = copy.deepcopy(self.avl[idx].city)
        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
        df = copy.deepcopy(self.avl[idx].seq_df)
        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))

        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1) # trajs [num_rows, 2]
        
        steps = [mapping[x] for x in df['TIMESTAMP'].values] # 时间的序列
        steps = np.asarray(steps, np.int64) # map from pandas order to sorted order

        # TRACK_ID和OBJECT_TYPE一起作为key来分组（两者都相同，则为同一组）
        # objs type: PrettyDict(similar to Dict)
        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
        keys = list(objs.keys()) # list of tuple(track, type)
        obj_type = [x[1] for x in keys] # list of type

        agt_idx = obj_type.index('AGENT') # group index
        idcs    = objs[keys[agt_idx]]
       
        agt_traj = trajs[idcs] # shape: ex. [50, 2]
        agt_step = steps[idcs]

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key] # get index of pandas table
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps
        return data
    
    def get_obj_feats(self, data):
        orig = data['trajs'][0][19].copy().astype(np.float32)

        if self.train and self.config['rot_aug']:
            theta = np.random.rand() * np.pi * 2.0
        else:
            pre = data['trajs'][0][18] - orig
            theta = np.pi - np.arctan2(pre[1], pre[0])

        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        feats, ctrs, gt_preds, has_preds, obs_trajs = [], [], [], [], []
        gt_local_preds = []
        for traj, step in zip(data['trajs'], data['steps']):
            if 19 not in step:
                continue

            gt_pred  = np.zeros((30, 2), np.float32)
            has_pred = np.zeros(30, np.bool)
            future_mask = np.logical_and(step >= 20, step < 50)
            post_step = step[future_mask] - 20 # future steps
            post_traj = traj[future_mask] # [30,2]
            
            gt_local_pred = np.zeros((30, 2), np.float32)
            gt_local_pred[post_step] = np.matmul(rot, (post_traj - orig.reshape(-1, 2)).T).T
            gt_local_preds.append(gt_local_pred)
            
            gt_pred[post_step] = post_traj # sorted [30,2]
            has_pred[post_step] = 1
            
            obs_mask = step < 20
            step = step[obs_mask]
            traj = traj[obs_mask]
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]
            
            for i in range(len(step)):
                if step[i] == 19 - (len(step) - 1) + i:
                    break
            step = step[i:]
            traj = traj[i:]

            feat = np.zeros((20, 3), np.float32)
            # R(world->ego) * Vec(world) = Vec(ego)
            feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat[step, 2]  = 1.0

            x_min, x_max, y_min, y_max = self.config['pred_range']
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue
            
            obs_trajs.append( copy.deepcopy(feat) )
            ctrs.append(feat[-1, :2].copy())
            feat[1:, :2] -= feat[:-1, :2] # driving direction
            feat[step[0], :2] = 0
            feats.append(feat)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, np.float32)
        ctrs  = np.asarray(ctrs, np.float32)
        gt_preds  = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)
        
        gt_local_preds = np.asarray(gt_local_preds, np.float32)
        obs_trajs = np.asarray(obs_trajs, np.float32)

        data['feats'] = feats         # [agents, seq=20, feat=3] direction
        data['ctrs']  = ctrs          # [agents, 2] 各agent的current node特征，在相对于AGENT的坐标系下
        data['orig']  = orig          # 每条数据中关心的预测：AGENT的地图坐标系位置，shape=[2,]
        data['theta'] = theta         # AGENT在current time上的angle
        data['rot']   = rot           # AGENT在current time上的angle对应的Rotation(world->ego)
        data['gt_preds']  = gt_preds  # [agents, 30, 2] 每个agent也有各自的gt traj
        data['has_preds'] = has_preds # [agents, 30, ]
        
        data['obs_trajs']      = obs_trajs # [agents, seq=20, feat=3] position
        data['gt_local_preds'] = gt_local_preds
        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = self.config['pred_range']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
        lane_ids = copy.deepcopy(lane_ids)
        #print("to process each land id.")
        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)
            # translate center line points from world_coord to ego_coord
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                # get polygon of this lane, shape [N,]
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

 
        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []

        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1
            
            # lane.centerline上点的数量 决定ctrs的size
            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32)) # segment center point
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32)) # segment direction
            
            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))
        
        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count # all segs (this seg is 'seg within lane' rather than lane_segment)
        
        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]
            idcs = node_idcs[i]
            
            pre['u'] += idcs[1:] # v -> u
            pre['v'] += idcs[:-1]
            if lane.predecessors is not None: #两个lane边界的处理，前者最后一个node和后者第一个node
                for nbr_id in lane.predecessors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])
                    
            suc['u'] += idcs[:-1] #  u -> v
            suc['v'] += idcs[1:]
            if lane.successors is not None:
                for nbr_id in lane.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])

        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.l_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.r_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)
                    
        graph = dict()
        # feature or relation for node
        graph['ctrs'] = np.concatenate(ctrs, 0) # center point coordinate of each node
        graph['num_nodes'] = num_nodes # num of segments 
        graph['feats'] = np.concatenate(feats, 0) # direction
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['pre'] = [pre] # relation between lane's seg(or node)
        graph['suc'] = [suc]
        graph['lane_idcs'] = lane_idcs # lane index(not lane_id) of node
        # relation between 2 lanes
        graph['pre_pairs'] = pre_pairs 
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs
        
        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)
        
        for key in ['pre', 'suc']:
            if 'scales' in self.config and self.config['scales']:
                #TODO: delete here
                graph[key] += dilated_nbrs2(graph[key][0], graph['num_nodes'], self.config['scales'])
            else:
                graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], self.config['num_scales']) # num_scales=6
        return graph


class ArgoTestDataset(ArgoDataset):
    def __init__(self, split, config, train=False):

        self.config = config
        self.train = train
        split2 = config['val_split'] if split=='val' else config['test_split']
        split = self.config['preprocess_val'] if split=='val' else self.config['preprocess_test']

        self.avl = ArgoverseForecastingLoader(split2)
        if 'preprocess' in config and config['preprocess']:
            if train:
                self.split = np.load(split, allow_pickle=True)
            else:
                self.split = np.load(split, allow_pickle=True)
        else:
            self.avl = ArgoverseForecastingLoader(split)
            self.am = ArgoverseMap()
            

    def __getitem__(self, idx):
        if 'preprocess' in self.config and self.config['preprocess']:
            data = self.split[idx]
            data['argo_id'] = int(self.avl.seq_list[idx].name[:-4]) #160547

            if self.train and self.config['rot_aug']:
                #TODO: Delete Here because no rot_aug
                new_data = dict()
                for key in ['orig', 'gt_preds', 'has_preds']:
                    new_data[key] = ref_copy(data[key])

                dt = np.random.rand() * self.config['rot_size']#np.pi * 2.0
                theta = data['theta'] + dt
                new_data['theta'] = theta
                new_data['rot'] = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)

                rot = np.asarray([
                    [np.cos(-dt), -np.sin(-dt)],
                    [np.sin(-dt), np.cos(-dt)]], np.float32)
                new_data['feats'] = data['feats'].copy()
                new_data['feats'][:, :, :2] = np.matmul(new_data['feats'][:, :, :2], rot)
                new_data['ctrs'] = np.matmul(data['ctrs'], rot)

                graph = dict()
                for key in ['num_nodes', 'turn', 'control', 'intersect', 'pre', 'suc', 'lane_idcs', 'left_pairs', 'right_pairs']:
                    graph[key] = ref_copy(data['graph'][key])
                graph['ctrs'] = np.matmul(data['graph']['ctrs'], rot)
                graph['feats'] = np.matmul(data['graph']['feats'], rot)
                new_data['graph'] = graph
                data = new_data
            else:
                new_data = dict()
                for key in ['orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph','argo_id','city']:
                    if key in data:
                        new_data[key] = ref_copy(data[key])
                data = new_data
            return data

        data = self.read_argo_data(idx)
        data = self.get_obj_feats(data)
        data['graph'] = self.get_lane_graph(data)
        data['idx'] = idx
        return data
    
    def __len__(self):
        if 'preprocess' in self.config and self.config['preprocess']:
            return len(self.split)
        else:
            return len(self.avl)

class MapQuery(object):
    #TODO: DELETE HERE No used
    """[Deprecated] Query rasterized map for a given region"""
    def __init__(self, scale, autoclip=True):
        """
        scale: one meter -> num of `scale` voxels 
        """
        super(MapQuery, self).__init__()
        assert scale in (1,2,4,8)
        self.scale = scale
        root_dir = '/mnt/yyz_data_1/users/ming.liang/argo/tmp/map_npy/'
        mia_map = np.load(f"{root_dir}/mia_{scale}.npy")
        pit_map = np.load(f"{root_dir}/pit_{scale}.npy")
        self.autoclip = autoclip
        self.map = dict(
            MIA=mia_map,
            PIT=pit_map
        )
        self.OFFSET = dict(
                MIA=np.array([502,-545]),
                PIT=np.array([-642,211]),
            )
        self.SHAPE=dict(
                MIA=(3674, 1482),
                PIT= (3043, 4259)
            )
    def query(self,region,theta=0,city='MIA'):
        """
        region: [x0,x1,y0,y1]
        city: 'MIA' or 'PIT'
        theta: rotation of counter-clockwise, angel/degree likd 90,180
        return map_mask: 2D array of shape (x1-x0)*scale, (y1-y0)*scale
        """
        region = [int(x) for x in region]

        map_data = self.map[city]
        offset = self.OFFSET[city]
        shape = self.SHAPE[city]
        x0,x1,y0,y1 = region
        x0,x1 = x0+offset[0],x1+offset[0]
        y0,y1 = y0+offset[1],y1+offset[1]
        x0,x1,y0,y1 = [round(_*self.scale) for _ in [x0,x1,y0,y1]]
        # extend the crop region to 2x -- for rotation
        H,W = y1-y0,x1-x0
        x0 -= int(round(W/2))
        y0 -= int(round(H/2))
        x1 += int(round(W/2))
        y1 += int(round(H/2))
        results = np.zeros([H*2,W*2])
        # padding of crop -- for outlier
        xstart,ystart=0,0
        if self.autoclip:
            if x0<0:
                xstart = -x0 
                x0 = 0
            if y0<0:
                ystart = -y0 
                y0 = 0
            x1 = min(x1,shape[1]*self.scale-1)
            y1 = min(y1,shape[0]*self.scale-1)
        map_mask = map_data[y0:y1,x0:x1]
        _H,_W = map_mask.shape
        results[ystart:ystart+_H, xstart:xstart+_W]=map_mask
        results = results[::-1] # flip to cartesian
        # rotate and remove margin
        rot_map = rotate(results,theta,center=None,order=0) # center None->map center
        H,W = results.shape
        outputH,outputW = round(H/2),round(W/2)
        startH,startW = round(H//4),round(W//4)
        crop_map = rot_map[startH:startH+outputH,startW:startW+outputW]
        return crop_map


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data

# num_scales: 6
def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), np.bool)
    # 生成mat, size=[num_nodes, num_nodes], nbr['u']和nbr['v']分别存着row id和col id.
    # 有len(data)==len(nbr['u'])==len(nbr['v']), 将data[k]的值(True)赋给mat[nbr['u'][k], nbr['v'][k]].
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), 
        shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales): # each scale consider one indirect relation (i-k-j)
        # if exist edge i-k and k-j, then give edge to i-j
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs


def dilated_nbrs2(nbr, num_nodes, scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, max(scales)):
        mat = mat * csr

        if i + 1 in scales:
            nbr = dict()
            coo = mat.tocoo()
            nbr['u'] = coo.row.astype(np.int64)
            nbr['v'] = coo.col.astype(np.int64)
            nbrs.append(nbr)
    return nbrs

 
def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


def cat(batch):
    if torch.is_tensor(batch[0]):
        batch = [x.unsqueeze(0) for x in batch]
        return_batch = torch.cat(batch, 0)
    elif isinstance(batch[0], list) or isinstance(batch[0], tuple):
        batch = zip(*batch)
        return_batch = [cat(x) for x in batch]
    elif isinstance(batch[0], dict):
        return_batch = dict()
        for key in batch[0].keys():
            return_batch[key] = cat([x[key] for x in batch])
    else:
        return_batch = batch
    return return_batch


def get_dist_np(offsets):
    if len(offsets) == 0:
        return 0.0
    return np.sum(np.sqrt(np.sum(np.square(offsets), axis=-1)))

# edge_mat: graph adj matrix
def get_lanes_with_dfs(
        edge_mat   : np.ndarray, 
        target_lid : int, 
        lane_idcs  : np.ndarray, 
        feats      : np.ndarray, 
        thres      = 60.0):
    num_lanes = len(edge_mat)
    mat = np.zeros((1, num_lanes), dtype=np.bool_)
    mat[0, target_lid] = 1
    lane_set = []
    mask     = (lane_idcs == target_lid)
    dist_sum = get_dist_np(feats[mask]) # get distance of target lane
    while True:
        if dist_sum >= thres:
            break
        mat  = np.matmul(mat, edge_mat)
        lids = np.nonzero(mat)[1]
        if len(lids) == 0:
            break
        dists = []
        for lid in lids:
            mask = (lane_idcs == lid)
            dists.append(get_dist_np(feats[mask]))
            lane_set.append(lid)
        dist_sum += min(dists)
    return lane_set

def fully_overlaped(lane_ids : np.ndarray, nbrs : np.ndarray):   
    for it in lane_ids:
        if it not in nbrs:     
            return False
    return True

def get_nbr_set(nbr_mat : np.ndarray, lanes : list()) -> np.ndarray:
    num_lanes = len(nbr_mat)
    mat  = np.zeros((1, num_lanes), dtype=np.bool_)
    nbrs = np.array(lanes) # lane ids
    mat[0, nbrs] = 1
    while True:
        mat = np.matmul(mat, nbr_mat)
        lane_ids = np.nonzero(mat)[1]
        if fully_overlaped(lane_ids, nbrs):
            break
        nbrs = np.unique(np.concatenate([nbrs, lane_ids]))
    return nbrs

def get_velocity_per_agent(agent_feats : np.ndarray, cycle_time=0.1):
    num_agents  = len(agent_feats)
    direct = np.sqrt((agent_feats[:,:,:2]**2).sum(-1))
    mask = direct > 0
    # duration = mask.sum(axis=1) * cycle_time # time
    
    increment = 0.1 * np.arange(mask.shape[1]) / mask.shape[1]
    last  = mask.astype(float) + increment 
    first = mask.astype(float) - increment
    last_val, last_idc = last.max(1), last.argmax(1)
    first_idc = first.argmax(1)
    duration = (last_idc - first_idc + 1) * cycle_time
    
    is_valid = last_val >= 1.0
    lon_velocity = np.zeros((num_agents), np.float32)
    # is_valid = duration > 0
    lon_distance = direct.sum(axis=1) # distance
    lon_velocity[is_valid] = (lon_distance[is_valid] / duration[is_valid])
    return lon_velocity, lon_distance


# AGs: agents, GNs: graph nodes
# input:
#   max_vel_threshold: 10 kph / 3.6 = 2.78 m/s
def generate_lane_roi(data, horizon_buffer = 20., max_vel_threshold = 2.78):
    obs_steps   = data['feats'].shape[1] # 20
    agent_feats = data['feats']     # [AGs, seq=20, feat=3]
    agent_ctrs  = data['ctrs']      # [AGs, 2]
    gt_preds    = data['gt_preds']  # [AGs, 30, 2]
    has_preds   = data['has_preds'] # [AGs, 30]
    graph       = data['graph']
    lane_idcs   = graph['lane_idcs'] # [GNs,]
    num_lanes   = lane_idcs[-1] + 1
    num_nodes   = len(lane_idcs)
    num_agents  = len(agent_ctrs)

    ## get distance between map node and agent, shape=[Gs, AGs, 2]
    dist = np.expand_dims(graph['ctrs'], axis=1) - np.expand_dims(agent_ctrs, axis=0)
    dist = np.sqrt((dist**2).sum(-1))
    sorted_nodes_idcs = dist.argsort(axis=0) # [Gs, AGs]
    
    '''
    sorted_node_dirs  = graph['feats'][sorted_nodes_idcs] # [Gs, AGs, 2]
    t1 = np.arctan2(agent_feats[:, -1, 1], agent_feats[:, -1, 0]).reshape(1, -1) # [1, AGs]
    t2 = np.arctan2(sorted_node_dirs[:, :, 1], sorted_node_dirs[:, :, 0]) # [Gs, AGs]
    dt = np.abs(t1 - t2)
    mask = dt > np.pi
    dt[mask] = np.abs(dt[mask] - 2 * np.pi)
    mask = dt < 0.25 * np.pi
    match_node_id_per_agent = [ sorted_nodes_idcs[mask[:,i], i][0] for i in range(num_agents) ]
    '''

    ## get interest nodes(<= 5.0) per agent
    closed_node_ids, closed_agent_ids = np.nonzero(dist < 5.0)
    assert len(gt_preds) == num_agents

    if True:
        pre   = np.zeros((num_lanes, num_lanes), dtype=np.bool_)
        suc   = np.zeros((num_lanes, num_lanes), dtype=np.bool_)
        left  = np.zeros((num_lanes, num_lanes), dtype=np.bool_)
        right = np.zeros((num_lanes, num_lanes), dtype=np.bool_)
        if len(graph['pre_pairs']) > 0:
            pre[ graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1] ] = 1
        if len(graph['suc_pairs']) > 0:
            suc[ graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1] ] = 1
        if len(graph['left']['u']) > 0:
            lane_ids_u = lane_idcs[graph['left']['u']]
            lane_ids_v = lane_idcs[graph['left']['v']]
            left[ lane_ids_u, lane_ids_v ] = 1
        if len(graph['right']['u']) > 0:
            lane_ids_u = lane_idcs[graph['right']['u']]
            lane_ids_v = lane_idcs[graph['right']['v']]
            right[ lane_ids_u, lane_ids_v ] = 1

    node_rel = dict()
    for k1 in ["pre", "suc"]:
        node_rel[k1] = []
        for ig in range(6):
            temp = np.zeros((num_nodes, num_nodes), np.bool_)
            temp[graph[k1][ig]['u'], graph[k1][ig]['v']] = 1
            node_rel[k1].append(temp) 
    for k1 in ["left", "right"]:
        temp = np.zeros((num_nodes, num_nodes), np.bool_)
        temp[graph[k1]['u'], graph[k1]['v']] = 1
        node_rel[k1] = temp
    
    vel_per_agent, _ = get_velocity_per_agent(agent_feats)
    sub_graph_per_agent = []
    valid_agent_ids = []
    
    for agent_idx in range(num_agents):
        if vel_per_agent[agent_idx] == 0: 
            continue
        
        ## get search horizon
        suc_horizon = vel_per_agent[agent_idx] * 3.0 + horizon_buffer # (3.0s trajectory + 20m)
        pre_horizon = vel_per_agent[agent_idx] * 2.0 + horizon_buffer # (2.0s trajectory + 20m)
        
        # find ctr node closest to agent_ctr and coresp. lane id
        cur_agt_dir = agent_feats[agent_idx, -1, :2]
        sorted_node_idcs_ = sorted_nodes_idcs[:, agent_idx]
        sorted_node_dirs  = graph['feats'][sorted_node_idcs_]
        t1 = np.arctan2(cur_agt_dir[1], cur_agt_dir[0])
        t2 = np.arctan2(sorted_node_dirs[:, 1], sorted_node_dirs[:, 0])
        dt = np.abs(t1 - t2)
        mask = dt > np.pi
        dt[mask] = np.abs(dt[mask] - 2 * np.pi)
        mask = dt < 0.25 * np.pi
        if len(sorted_node_idcs_[mask]) == 0:
            mask = dt < 0.5 * np.pi
            if len(sorted_node_idcs_[mask]) == 0:
                continue
        node_id = sorted_node_idcs_[mask][0]
        

        # find pre and suc lanes
        target_lane_id = lane_idcs[node_id]
        searched_lane_ids = [target_lane_id]
        searched_lane_ids.extend( get_lanes_with_dfs(suc, target_lane_id, lane_idcs, graph['feats'], suc_horizon) )
        searched_lane_ids.extend( get_lanes_with_dfs(pre, target_lane_id, lane_idcs, graph['feats'], pre_horizon) )
        roi_lane_idcs = get_nbr_set(left + right, searched_lane_ids) # find left and right lanes

        # create lane roi for each agent:
        sub_graph  = dict()
        node_mask  = np.concatenate([ np.nonzero(lane_idcs == x)[0]
            for x in roi_lane_idcs ])
        if len(node_mask) < 6: # ignore this agent who has too less nodes
            continue
        
        ## extract map node feature from global graph
        sub_graph_feats = np.zeros((len(node_mask), 8), dtype=np.float32)
        sub_graph_feats[:, :2]  = graph['ctrs'][node_mask]  # [node', 2]
        sub_graph_feats[:, 2:4] = graph['feats'][node_mask] # [node', 2]
        sub_graph_feats[:, 4:6] = graph['turn'][node_mask]  # [node', 2]
        sub_graph_feats[:, 6]   = graph['control'][node_mask]
        sub_graph_feats[:, 7]   = graph['intersect'][node_mask]

        motion_feat = np.concatenate([
            data['obs_trajs'][agent_idx, :, :2],
            data['feats'][agent_idx, :, :2],
        ], axis=-1) # [20, 4]
        interest_node_idcs = closed_node_ids[closed_agent_ids==agent_idx]

        # get index of sub graph (node_mask store index of global graph)
        associated_node_idcs = [i for i, nid in enumerate(node_mask) if nid in interest_node_idcs ]

        vs = np.array(associated_node_idcs, dtype=np.int32)
        us = np.zeros(len(vs), dtype=np.int32)
        sub_graph['a2m']        = {'u': us, 'v': vs}
        sub_graph['node_mask']  = node_mask
        sub_graph['num_nodes']  = len(node_mask)
        sub_graph['feats']      = sub_graph_feats
        sub_graph['agent_feat'] = motion_feat.reshape(-1) # [20 * 4]
        sub_graph['agent_vel']  = vel_per_agent[agent_idx]

        for k1 in ["pre", "suc"]:
            sub_graph[k1] = []
            for ig in range(6):
                temp   = node_rel[k1][ig] # relation matrix
                us, vs = np.nonzero(temp[node_mask][:, node_mask])
                assert isinstance(us, np.ndarray) and isinstance(vs, np.ndarray)
                sub_graph[k1].append({'u': us, 'v': vs})
        
        ## ignore this agent if pre/suc_0 is empty
        if len(sub_graph["pre"][0]["u"]) == 0 and len(sub_graph["suc"][0]["u"]) == 0:
            continue
        
        for k1 in ["left", "right"]:
            temp   = node_rel[k1] # relation matrix
            us, vs = np.nonzero(temp[node_mask][:, node_mask])
            assert isinstance(us, np.ndarray) and isinstance(vs, np.ndarray)
            sub_graph[k1] = {'u': us, 'v': vs}
        
        sub_graph_per_agent.append(sub_graph)
        valid_agent_ids.append(agent_idx)
    
    data["subgraphs"]       = sub_graph_per_agent
    data["valid_agent_ids"] = np.asarray(valid_agent_ids, np.int16)
    return data
