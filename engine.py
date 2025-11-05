import torch
from typing import Iterable, Optional
from timm.utils import accuracy, dispatch_clip_grad
import time
from torch_cluster import radius_graph
import torch_geometric
from spectrum.loss import gmm_loss, fc_loss
import numpy as np
import itertools

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_step(model: torch.nn.Module, criterion: torch.nn.Module,
                    norm_factor: list, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    loss_scaler=None,
                    clip_grad=None,
                    input_step=0,
                    print_freq: int = 100, 
                    loss_type = 'MAE',
                    spec_type = 'FC',
                    line_shape = 'gaussian',
                    beta = 2.0,
                    logger=None):
    
    model.train()
    criterion.train()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    
    start_time = time.perf_counter()
    
    task_mean = norm_factor[0] #model.task_mean
    task_std  = norm_factor[1] #model.task_std

    #atomref = dataset.atomref()
    #if atomref is None:
    #    atomref = torch.zeros(100, 1)
    #atomref = atomref.to(device)
    
    for step, data in enumerate(data_loader):
        data = data.to(device)
        pred = model(f_in=data.x, pos=data.pos, batch=data.batch, 
            node_atom=data.z,
            edge_d_index=data.edge_d_index, edge_d_attr=data.edge_d_attr)
        pred = pred.squeeze()
        pred = pred*task_std + task_mean #unnormalization
        if spec_type == 'Naive':
            loss = criterion(pred, data.y)
        elif spec_type == 'GMM':
            loss = gmm_loss(data.spec_x, data.spec_y, pred, loss_type=loss_type)
        elif spec_type == 'FC':
            loss = fc_loss(data.spec_x, data.spec_y, pred, loss_type=loss_type, line_shape=line_shape, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad is not None:
            dispatch_clip_grad(model.parameters(), 
                value=clip_grad, mode='norm')
        optimizer.step()
        
        loss_metric.update(loss.item(), n=pred.shape[0])
        err = pred.detach() - data.y
        mae_metric.update(torch.mean(torch.abs(err)).item(), n=pred.shape[0])
        
        torch.cuda.synchronize()

        # logging
        if input_step % print_freq == 0:
            info_str = 'Step: [{input_step}] \t loss: {loss:.5f}, MAE: {mae:.5f} '.format( 
                input_step=input_step, 
                mae=mae_metric.avg, 
                loss=loss_metric.avg,
                )
            info_str += 'lr={:.2e}'.format(optimizer.param_groups[0]["lr"])
            logger.info(info_str)
        
    return mae_metric.avg


def evaluate(model, norm_factor, data_loader, device, print_freq=100, 
        loss_type = 'MAE',
        spec_type = 'FC',
        line_shape = 'gaussian',
        beta = 2.0,
        logger=None):
    
    model.eval()
    
    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    criterion = torch.nn.L1Loss()
    criterion.eval()
    
    task_mean = norm_factor[0] #model.task_mean
    task_std  = norm_factor[1] #model.task_std

    id_buffer = []
    pred_buffer: list[torch.Tensor] = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            pred = model(f_in=data.x, pos=data.pos, batch=data.batch, 
                node_atom=data.z,
                edge_d_index=data.edge_d_index, edge_d_attr=data.edge_d_attr)
            pred = pred.squeeze()
            pred = pred*task_std + task_mean #unnormalization

            id_buffer.append(data['name'])
            pred_buffer.append(pred.cpu())
            
            if spec_type == 'Naive':
                loss = criterion(pred, data.y)
            elif spec_type == 'GMM':
                loss = gmm_loss(data.spec_x, data.spec_y, pred, loss_type=loss_type)
            elif spec_type == 'FC':
                loss = fc_loss(data.spec_x, data.spec_y, pred, loss_type=loss_type, line_shape=line_shape, beta=beta)
            loss_metric.update(loss.item(), n=pred.shape[0])
            err = pred.detach() - data.y
            mae_metric.update(torch.mean(torch.abs(err)).item(), n=pred.shape[0])

    preds = torch.cat(pred_buffer, 0)
    ids = list(itertools.chain.from_iterable(id_buffer))
        
    return mae_metric.avg, loss_metric.avg, preds, ids


def compute_stats(data_loader, max_radius, logger, print_freq=1000):
    '''
        Compute mean of numbers of nodes and edges
    '''
    log_str = '\nCalculating statistics with '
    log_str = log_str + 'max_radius={}\n'.format(max_radius)
    logger.info(log_str)
        
    avg_node = AverageMeter()
    avg_edge = AverageMeter()
    avg_degree = AverageMeter()
    
    for step, data in enumerate(data_loader):
        
        pos = data.pos
        batch = data.batch
        edge_src, edge_dst = radius_graph(pos, r=max_radius, batch=batch,
            max_num_neighbors=1000)
        batch_size = float(batch.max() + 1)
        num_nodes = pos.shape[0]
        num_edges = edge_src.shape[0]
        num_degree = torch_geometric.utils.degree(edge_src, num_nodes)
        num_degree = torch.sum(num_degree)
            
        avg_node.update(num_nodes / batch_size, batch_size)
        avg_edge.update(num_edges / batch_size, batch_size)
        avg_degree.update(num_degree / (num_nodes), num_nodes)
            
        if step % print_freq == 0 or step == (len(data_loader) - 1):
            log_str = '[{}/{}]\tavg node: {}, '.format(step, len(data_loader), avg_node.avg)
            log_str += 'avg edge: {}, '.format(avg_edge.avg)
            log_str += 'avg degree: {}, '.format(avg_degree.avg)
            logger.info(log_str)
