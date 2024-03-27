from __future__ import division, absolute_import
import math
import os
import time
import torch
seed=123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import argparse
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import scipy
from sklearn.preprocessing import normalize
import torch.nn as nn
from models.SVCNN import  Autoencoder
import torch.nn.functional as F
from losses.center_loss import CrossModalCenterLoss
from losses.cross_modal_loss import CrossModalLoss
from losses.paws import paws_loss
from losses.zinb_loss import ZINB
#from losses.scl import MGLoss,SupConLoss,modal_alignment_loss
from sc_dataset import snRNA_snmC,snRNA_snATAC,CITE_ASAP
from torch.utils.data import ConcatDataset
import warnings
warnings.filterwarnings("ignore")

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_features_and_labels(loader,model_rna,model_atac):
    model_rna.eval()
    model_atac.eval()
    labels, rna_feats, atac_feats = [], [], []
    with torch.no_grad():
        for minibatch in loader:          
            rnas = minibatch[0].to(torch.float32)
            atacs = minibatch[1].to(torch.float32)
            gts = minibatch[2].to(torch.long)
            rnas = rnas.cuda(non_blocking=True)
            atacs = atacs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            rna_feat,_ = model_rna(rnas)               
            atac_feat,_ = model_atac(atacs)
            rna_feats.append(rna_feat)
            atac_feats.append(atac_feat)
            labels.append(gts)
    model_rna.train()
    model_atac.train()
    return torch.hstack(labels), torch.vstack(rna_feats), torch.vstack(atac_feats)

def training(args):
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.dataset == 'CITE_ASAP':
        labeled_trainset = CITE_ASAP('train',args.n_labeled, labeled_dataset = True)
        unlabeled_trainset = CITE_ASAP('train',args.n_labeled, labeled_dataset = False)
        num_classes = 7
    elif args.dataset == 'snRNA_snATAC':      
        labeled_trainset = snRNA_snATAC('train',args.n_labeled, labeled_dataset = True)
        unlabeled_trainset = snRNA_snATAC('train',args.n_labeled, labeled_dataset = False)
        num_classes = 18
    elif args.dataset == 'snRNA_snmC':
        labeled_trainset = snRNA_snmC('train',args.n_labeled, labeled_dataset = True)
        unlabeled_trainset = snRNA_snmC('train',args.n_labeled, labeled_dataset = False) 
        num_classes = 17
        
    print('dataset:', args.dataset, 'labeled length:', len(labeled_trainset),'unlabeled length:', len(unlabeled_trainset), 'classes:', num_classes)

    model_rna = Autoencoder(args.dataset)
    model_atac = Autoencoder(args.dataset)
        
    sup_criterion = nn.CrossEntropyLoss(reduction='mean')
    #unsup_criterion = nn.CrossEntropyLoss(reduction='mean')
    cmc_criterion = CrossModalCenterLoss(num_classes=num_classes)
    # mse_criterion = nn.MSELoss()
    # align_criterion = MGLoss(temperature=0.07)
    mg_criterion = CrossModalLoss()
    paws_criterion = paws_loss()
    
    optimizer_rna = optim.Adam(model_rna.parameters(), lr=args.lr_rna, weight_decay=args.weight_decay)
    optimizer_atac = optim.Adam(model_atac.parameters(), lr=args.lr_atac, weight_decay=args.weight_decay)
    optimizer_centloss = optim.SGD(cmc_criterion.parameters(), lr=args.lr_center)
    
    train_loader = torch.utils.data.DataLoader(labeled_trainset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=False,pin_memory=True)
    unsupervised_train_loader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=False,pin_memory=True)
    combined_trainset = ConcatDataset([labeled_trainset, unlabeled_trainset])
    pretrain_loader = torch.utils.data.DataLoader(combined_trainset, batch_size=200, shuffle=True, num_workers=10, drop_last=False,pin_memory=True)
    iteration = 0

    num_train_rnas = len(labeled_trainset)
    num_unsup_rnas = len(unlabeled_trainset)
    max_samples = max(num_train_rnas, num_unsup_rnas)     # Define the iterations in an epoch
    niters_per_epoch = int(math.ceil(max_samples * 1.0 // args.batch_size))
    print('iterations_per_epoch:',niters_per_epoch)
    best = 0

    optimizer1 = torch.optim.Adam(model_rna.parameters(),lr=0.0001)
    optimizer2 = torch.optim.Adam(model_atac.parameters(),lr=0.0001)
    criterion = ZINB()
    
    model_rna.train(True).to('cuda')
    model_atac.train(True).to('cuda')
    for epoch in range(args.pretrain_epochs):
        for batch_idx, data in enumerate(pretrain_loader):
            rna,atac = data[0].cuda(),data[1].cuda()
            pi, disp, mean = model_rna.pretrain_forward(rna)
            loss1 = criterion(pi, disp, rna, mean) + 0.1 * F.mse_loss(mean, rna)
            #loss1 = F.mse_loss(mean, rna)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            
            pi, disp, mean = model_atac.pretrain_forward(atac)
            loss2 = criterion(pi, disp, atac, mean) + 0.1 * F.mse_loss(mean, atac) 
            #loss2 = F.mse_loss(mean, atac)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()        
        print('Pretrain epoch [{}/{}], ZINB loss rna:{:.4f}, ZINB loss atac:{:.4f}'.format(epoch+1, args.pretrain_epochs, loss1.item(), loss2.item()))
    
    # bulid memory bank for rna/atac features and scores
    num_sample = len(unlabeled_trainset) 
    rna_fea_bank = torch.randn(num_sample, 256)
    rna_score_bank = torch.randn(num_sample, num_classes).cuda()
    atac_fea_bank = torch.randn(num_sample, 256)
    atac_score_bank = torch.randn(num_sample, num_classes).cuda()
    model_rna.eval()
    model_atac.eval()
    with torch.no_grad():
        iter_test = iter(unsupervised_train_loader)
        for i in range(len(unsupervised_train_loader)):
            data = iter_test.__next__()
            rna,atac,indx = data[0].cuda(),data[1].cuda(),data[-1]
            f_rna,p_rna = model_rna(rna)  
            f_atac,p_atac = model_atac(atac)
            f_rna_norm = F.normalize(f_rna)
            f_atac_norm = F.normalize(f_atac)
            rna_outputs = nn.Softmax(-1)(p_rna)
            atac_outputs = nn.Softmax(-1)(p_atac)
            rna_fea_bank[indx] = f_rna_norm.detach().clone().cpu()
            rna_score_bank[indx] = rna_outputs.detach().clone()  
            atac_fea_bank[indx] = f_atac_norm.detach().clone().cpu()
            atac_score_bank[indx] = atac_outputs.detach().clone() 
    # end of buliding memory bank
    
    model_rna.train()
    model_atac.train()
    for epoch in range(args.epochs):
        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)
        # ## generate prototypes with labeled data by confidence voting every epoch
        # labels,rna_feats,atac_feats = get_features_and_labels( train_loader,model_rna,model_atac)
        # # rna_prototypes = torch.nan_to_num(torch.vstack([rna_feats[labels == i].mean(dim=0) for i in range(num_classes)])) ##center shape: (num_classes, dim) -->(10,128)
        # # atac_prototypes = torch.nan_to_num(torch.vstack([atac_feats[labels == i].mean(dim=0) for i in range(num_classes)]))
        # # print('labels: ', labels.shape, 'rna_feats: ',rna_feats.shape,'atac_feats: ', atac_feats.shape)
        # M = 1
        # intra_rna_feats = {}
        # intra_atac_feats = {}
        # for label, fi, fp in zip(labels, rna_feats, atac_feats):
        #     if label.item() not in intra_rna_feats:
        #         intra_rna_feats[label.item()] = []
        #     if label.item() not in intra_atac_feats:
        #         intra_atac_feats[label.item()] = []
                
        #     intra_rna_feats[label.item()].append(fi)
        #     intra_atac_feats[label.item()].append(fp)
            
        # rna_prototypes = []
        # atac_prototypes = []
        # rna_prototype_labels = []
        # atac_prototype_labels = []

        # for label, features in intra_rna_feats.items():
        #     features_tensor = torch.stack(features)
        #     rna_prototypes.append(features_tensor.mean(0))
        #     rna_prototype_labels.append(label)
        # rna_prototypes = torch.stack(rna_prototypes)
        # rna_prototype_labels = torch.tensor(rna_prototype_labels)

        # for label, features in intra_atac_feats.items():
        #     features_tensor = torch.stack(features)
        #     atac_prototypes.append(features_tensor.mean(0))
        #     atac_prototype_labels.append(label)
        # atac_prototypes = torch.stack(atac_prototypes)
        # atac_prototype_labels = torch.tensor(atac_prototype_labels)
        
        # # print('rna prototypes:', rna_prototypes.shape)
        # # print('rna prototype labels:', rna_prototype_labels.shape)
        # # print('atac prototypes:', atac_prototypes.shape)
        # # print('atac prototype labels:', atac_prototype_labels.shape)
            
        # rna_prototypes = torch.tensor(rna_prototypes).cuda().requires_grad_(False)
        # rna_prototype_labels = torch.tensor(rna_prototype_labels).cuda().requires_grad_(False)
        # atac_prototypes = torch.tensor(atac_prototypes).cuda().requires_grad_(False)
        # atac_prototype_labels = torch.tensor(atac_prototype_labels).cuda().requires_grad_(False)

        # ##momentum update centers
        # uni_prototypes = nn.Parameter(0.5*(rna_prototypes + atac_prototypes)).cuda()
        # cmc_criterion.centers.data = cmc_criterion.centers.data.mul(1-args.alpha).add_(args.alpha*uni_prototypes.data)
        
        
    ### data preparation 
        for idx in range(niters_per_epoch):
            start_time = time.time()
            optimizer_rna.zero_grad()
            optimizer_atac.zero_grad()
            optimizer_centloss.zero_grad()
            try:
                minibatch = dataloader.__next__()
                unsup_minibatch = unsupervised_dataloader.__next__()

            except:
                dataloader = iter(train_loader)
                unsupervised_dataloader = iter(unsupervised_train_loader)
        
                minibatch = dataloader.__next__()
                unsup_minibatch = unsupervised_dataloader.__next__()

            rnas = minibatch[0].to(torch.float32)
            atacs = minibatch[1].to(torch.float32)
            gts = minibatch[2].to(torch.long)
            unsup_rnas = unsup_minibatch[0].to(torch.float32)
            unsup_atacs = unsup_minibatch[1].to(torch.float32)
            unsup_idx = unsup_minibatch[-1]
            rnas = rnas.cuda(non_blocking=True)
            atacs = atacs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            unsup_rnas = unsup_rnas.cuda(non_blocking=True)
            unsup_atacs = unsup_atacs.cuda(non_blocking=True)
            #unsup_gts = unsup_minibatch[2].to(torch.long).cuda(non_blocking=True)
            ##supervised part
            sup_rna_feat,sup_rna_pred = model_rna(rnas)
            sup_atac_feat,sup_atac_pred = model_atac(atacs)

            # supervised cls loss    
            rna_sup_loss = sup_criterion(sup_rna_pred, gts)
            atac_sup_loss = sup_criterion(sup_atac_pred, gts)
            sup_cls_loss = rna_sup_loss + atac_sup_loss
            
            # supervised algin loss
            
            sup_align_loss = cmc_criterion(torch.cat((sup_rna_feat,sup_atac_feat), dim = 0), torch.cat((gts,gts), dim = 0)) 
            #sup_align_loss = align_criterion(torch.cat((sup_rna_feat.unsqueeze(1), sup_atac_feat.unsqueeze(1)),dim = 1),gts,gts).mean()

            # unsupervised part
            # with torch.no_grad():
            #     unsup_rna_feat,unsup_rna_pred = model_rna(unsup_rnas)
            #     unsup_atac_feat,unsup_atac_pred = model_atac(unsup_atacs)
            #     unsup_rna_pred = unsup_rna_pred.detach()
            #     unsup_atac_pred = unsup_atac_pred.detach()
                
            #     _, pseudo_rna_labels = torch.max(unsup_rna_pred, dim=1)
            #     pseudo_rna_labels = pseudo_rna_labels.long()
            #     _, pseudo_atac_labels = torch.max(unsup_atac_pred, dim=1)
            #     pseudo_atac_labels = pseudo_atac_labels.long()
            #     # print('pseudo_rna_labels:',pseudo_rna_labels)
            #     # print('pseudo_atac_labels:',pseudo_atac_labels)
            #     # print('unsup_labels:',unsup_gts)
            
            unsup_rna_feat,unsup_rna_pred = model_rna(unsup_rnas)
            unsup_atac_feat,unsup_atac_pred = model_atac(unsup_atacs)
                
            ## paws loss
            gts = F.one_hot(gts,num_classes).float()
            loss_paws = paws_criterion(unsup_rna_feat,sup_rna_feat,gts,unsup_atac_feat,sup_atac_feat,gts)
            
            softmax_rna = nn.Softmax(dim=1)(unsup_rna_pred)
            softmax_atac = nn.Softmax(dim=1)(unsup_atac_pred)
            
            with torch.no_grad():
                rna_feat = F.normalize(unsup_rna_feat)
                atac_feat = F.normalize(unsup_atac_feat)
                output_f_ = torch.cat([rna_feat,atac_feat], dim = 0).cpu().detach().clone() # 2*batch x dim
                
                rna_fea_bank[unsup_idx] = rna_feat.cpu().detach().clone()
                atac_fea_bank[unsup_idx] = atac_feat.cpu().detach().clone()
                fea_bank = torch.cat([rna_fea_bank,atac_fea_bank], dim = 0)  # 2*n x dim

                rna_score_bank[unsup_idx] =softmax_rna.detach().clone()
                atac_score_bank[unsup_idx] = softmax_atac.detach().clone()
                score_bank = torch.cat([rna_score_bank,atac_score_bank], dim = 0) # 2*n x num_classes
        
                distance = output_f_ @ fea_bank.T   # 2*batch x 2*n
                _, idx_near = torch.topk(distance, dim=-1,largest=True,k=args.K + 1)
                idx_near = idx_near[:, 1:]  # 2*batch x K
                score_near = score_bank[idx_near]  # 2*batch x K x C
                fea_near = fea_bank[idx_near]  # 2*batch x K x num_dim       
                fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1,  -1)  # 2*batch x n x dim
                distance_ = torch.bmm(fea_near,fea_bank_re.permute(0, 2, 1))  # 2*batch x K x n
                _, idx_near_near = torch.topk(distance_, dim=-1, largest=True, k=args.KK + 1)  # M near neighbors for each of above K ones  M = args.KK
                idx_near_near = idx_near_near[:, :, 1:]  # 2*batch x K x M
                unsup_idx_ = unsup_idx.unsqueeze(-1).unsqueeze(-1)
                unsup_idx_ = torch.cat([unsup_idx_,unsup_idx_], dim = 0)
                match = (idx_near_near == unsup_idx_).sum(-1).float()  # 2*batch x K
                weight = torch.where( match > 0., match, torch.ones_like(match).fill_(0))  # 2*batch x K     match > 0 -> weight = 1; match <= 0 -> weight = 0
                weight_kk = weight.unsqueeze(-1).expand(-1, -1,args.KK)  # 2*batch x K x M
                score_near_kk = score_bank[idx_near_near]  # 2*batch x K x M x C
                #print(weight_kk.shape)
                weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],-1)  # 2*batch x KM
                weight_kk = weight_kk.fill_(1)   # weight_kk = 1
                score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1, num_classes)  # 2*batch x KM x C

            ##nn of nn
            output_re_rna = softmax_rna.unsqueeze(1).expand(-1, args.K * args.KK,-1)  # batch x KM x C
            output_re_atac = softmax_atac.unsqueeze(1).expand(-1, args.K * args.KK, -1)  # batch x KM x C
            output_re = torch.cat([output_re_rna, output_re_atac],dim=0)
            const = torch.mean((F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) * weight_kk.cuda()).sum(1))   # 2*batch x KM x C ->  2*batch x KM -> 2*batch
            unsup_loss_KM = torch.mean(const)  

            unsup_loss_KM.backward(retain_graph=True)
            ## nn
            softmax_out_un_rna = softmax_rna.unsqueeze(1).expand(-1, args.K, -1)  # batch x K x C
            softmax_out_un_atac = softmax_atac.unsqueeze(1).expand(-1, args.K, -1)  # batch x K x C
            softmax_out_un = torch.cat([softmax_out_un_rna, softmax_out_un_atac],dim=0) # 2*batch x K x C
            unsup_loss_K = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))  #
                        
            # softmax_out = torch.cat([softmax_rna, softmax_atac],dim=0)
            # msoftmax = softmax_out.mean(dim=0)
            # im_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            
            # unsup_loss = unsup_loss_K + unsup_loss_KM + im_div
            
            unsup_loss = unsup_loss_K

            # pseudo labels
            # rna_dist = torch.cdist(F.normalize(unsup_rna_feat), F.normalize(rna_prototypes))
            # _, rna_topk_indices = rna_dist.topk(M, dim=1, largest=False)
            # topk_rna_labels = rna_prototype_labels[rna_topk_indices]
            # pseudo_rna_labels, _ = topk_rna_labels.mode(dim=1)
            
            # rna_predict = unsup_rna_feat @ rna_prototypes.T
            # rna_score = nn.Softmax(dim=-1)(rna_predict)
            # #print('rna_score:', rna_score)
            
            # atac_dist = torch.cdist(F.normalize(unsup_atac_feat), F.normalize(atac_prototypes))
            # _, atac_topk_indices = atac_dist.topk(M, dim=1, largest=False)
            # topk_atac_labels = atac_prototype_labels[atac_topk_indices]
            # pseudo_atac_labels, _ = topk_atac_labels.mode(dim=1)

            # atac_predict = unsup_atac_feat @ atac_prototypes.T
            # atac_score = nn.Softmax(dim=-1)(atac_predict)
            # #print('atac_score:', atac_score)
            
            # unsupervised cls loss
            # rna_unsup_loss = unsup_criterion(unsup_rna_pred, pseudo_rna_labels)
            # atac_unsup_loss = unsup_criterion(unsup_atac_pred, pseudo_atac_labels)
            # unsup_cls_loss = rna_unsup_loss + atac_unsup_loss

            # unsupervised algin loss
            # unsup_align_loss = cmc_criterion(torch.cat((unsup_rna_feat,unsup_atac_feat), dim = 0), torch.cat((pseudo_rna_labels,pseudo_atac_labels), dim = 0),
            #                                  torch.cat((softmax_rna,softmax_atac), dim = 0)) \
            #                                    + 0.1*mse_criterion(unsup_rna_feat, unsup_atac_feat)
            
            unsup_align_loss = mg_criterion(torch.cat((unsup_rna_feat,unsup_atac_feat), dim = 0))
            #unsup_align_loss = mg_criterion(torch.cat((unsup_rna_feat,unsup_atac_feat), dim = 0), torch.cat((softmax_rna,softmax_atac), dim = 0))
            #unsup_align_loss = cmc_criterion(torch.cat((unsup_rna_feat,unsup_atac_feat), dim = 0), torch.cat((pseudo_rna_labels,pseudo_atac_labels), dim = 0)) 
            #unsup_align_loss = mse_criterion(unsup_rna_feat, unsup_atac_feat)
            
            #total loss
            #unsup_loss = torch.tensor(0).cuda()
            loss = sup_cls_loss + sup_align_loss +  unsup_align_loss + unsup_loss + loss_paws
            #loss = sup_cls_loss + sup_align_loss 
            
            loss.backward()
            optimizer_rna.step()
            optimizer_atac.step()           
            optimizer_centloss.step()
            
            # for param in cmc_criterion.parameters():
            #     param.grad.data *= (1. / args.weight_center)
                
            if (iteration%args.lr_step) == 0:
                lr_rna = args.lr_rna * (0.1 ** (iteration // args.lr_step))
                lr_atac = args.lr_atac * (0.1 ** (iteration // args.lr_step))     
                lr_center = args.lr_center * (0.1 ** (iteration // args.lr_step))
                
                for param_group in optimizer_rna.param_groups:
                    param_group['lr_rna'] = lr_rna
                for param_group in optimizer_atac.param_groups:
                    param_group['lr_atac'] = lr_atac
                for param_group in optimizer_centloss.param_groups:
                    param_group['lr'] = lr_center
            


            if iteration % args.per_print == 0:
                print('[%d][%d] time: %f vid: %d' % (epoch, iteration, time.time() - start_time, gts.size(0)))
                print("sup_cls_loss: %f sup_align_loss: %f unsup_align_loss: %f unsup_loss k: %f paws_loss: %f" % 
                      (sup_cls_loss.item(),sup_align_loss.item(),unsup_align_loss.item(),unsup_loss.item(),loss_paws.item()))
                
                start_time = time.time()

            iteration = iteration + 1

            # if  (iteration%niters_per_epoch) == 0:
            #     print('----------------- Save The Network ------------------------')
            #     with open(args.save + str(args.n_labeled) + '-model_rna.pkl', 'wb') as f:
            #         torch.save(model_rna, f)
            #     with open(args.save + str(args.n_labeled) + '-model_atac.pkl', 'wb') as f:
            #         torch.save(model_atac, f)
            
        extract(model_rna,model_atac,args)
        res = eval_func(1)
        if res['average'] > best:
            best = res['average']
            ans = res
        print(ans)
    return ans

def extract(model_rna,model_atac,args):
    # model_rna = torch.load('%s/%d-model_rna.pkl' % (args.save, args.n_labeled),
    #                      map_location=lambda storage, loc: storage)
    model_rna = model_rna.eval()
    # model_atac = torch.load('%s/%d-model_atac.pkl' % (args.save, args.n_labeled),
    #                    map_location=lambda storage, loc: storage)
    model_atac = model_atac.eval()

    torch.cuda.empty_cache()
    #################################
  
    if args.dataset == 'CITE_ASAP':
        test_set = CITE_ASAP('test')
    elif args.dataset == 'snRNA_snATAC':
        test_set = snRNA_snATAC('test')
    elif args.dataset == 'snRNA_snmC':
        test_set = snRNA_snmC('test')    
        
    data_loader_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10)
    print('length of the dataset: ', len(test_set))
    #################################
    
    rna_feat_list = np.zeros((len(test_set), 256))
    atac_feat_list = np.zeros((len(test_set), 256))
    label = np.zeros((len(test_set)))
    #################################
    iteration = 0
    for data in data_loader_loader:
        rna_feat, atac_feat, ori_label = data[0],data[1],data[2]
            
        rna_feat = Variable(rna_feat).to(torch.float32).to('cuda')
        atac_feat = Variable(atac_feat).to(torch.float32).to('cuda')
        ori_label = Variable(ori_label).to(torch.long).to('cuda')
        ##########################################
        model_rna = model_rna.to('cuda')
        _rna_feat,_ = model_rna(rna_feat)
        model_atac = model_atac.to('cuda')
        _atac_feat,_ = model_atac(atac_feat)

        rna_feat_list[iteration, :] = _rna_feat.data.cpu().numpy()
        atac_feat_list[iteration, :] = _atac_feat.data.cpu().numpy()
        label[iteration] = ori_label.data.cpu().numpy()
        iteration = iteration + 1
            
    model_rna.train()
    model_atac.train()
    
    np.save('%s/%d-rna_feat' % (args.save, args.n_labeled), rna_feat_list)
    np.save('%s/%d-atac_feat' % (args.save, args.n_labeled), atac_feat_list)
    np.save('%s/%d-label' % (args.save, args.n_labeled), label)

    np.save('/home/zf/ai4science/cvpr24/sc_matching/vis/RNA/%s/%d-Ours_codes' % (args.dataset, args.n_labeled), rna_feat_list)
    np.save('/home/zf/ai4science/cvpr24/sc_matching/vis/ATAC/%s/%d-Ours_codes' % (args.dataset, args.n_labeled), atac_feat_list)
    np.save('/home/zf/ai4science/cvpr24/sc_matching/vis/RNA/%s/%d-label' % (args.dataset, args.n_labeled), label)
    
def fx_calc_map_label(view_1, view_2, label_test):
    dist = scipy.spatial.distance.cdist(view_1, view_2, 'cosine')  # rows view_1 , columns view 2
    ord = dist.argsort()
    numcases = dist.shape[0]
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(numcases):
            if label_test[i] == label_test[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)

def eval_func(rna_pairs):
    print('number of rna views: ', rna_pairs)
    rna_feat = np.load('%s/%d-rna_feat.npy' % (args.save, args.n_labeled))
    atac_feat = np.load('%s/%d-atac_feat.npy' % (args.save, args.n_labeled))
    label = np.load('%s/%d-label.npy' % (args.save, args.n_labeled))
    ########################################
    rna_test = normalize(rna_feat, norm='l1', axis=1)
    atac_test = normalize(atac_feat, norm='l1', axis=1)
    ########################################
    par_list = [
        (rna_test, atac_test, 'RNA2ATAC'),
        (atac_test, rna_test, 'ATAC2RNA')]
    ########################################
    
    name1 = ['RNA2ATAC', 'ATAC2RNA']
    res ={}
    avg_acc=0
    for index in range(2):
        view_1, view_2, name = par_list[index]
        print(name + '---------------------------')
        acc = fx_calc_map_label(view_1, view_2, label)
        
        acc_round = round(acc * 100, 2)
        print(str(acc_round))
        avg_acc+=acc
        res[name1[index]] = str(acc_round)
    avg_acc = round(avg_acc*100/2,2)
    res['average'] = avg_acc
    print('average---------------------------')
    print(str(avg_acc))         

    return res
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')

    parser.add_argument('--M', type=int, default=1, help='number of prototypes per class')
    
    parser.add_argument('--K', type=int, default=5)
    
    parser.add_argument('--KK', type=int, default=5)
    
    parser.add_argument('--alpha', type=float, default=0.01, help='momentum update rate')
    
    parser.add_argument('--warmup_epoch', type=int, default=0, help='warm up epoch')
    
    parser.add_argument('--gpu_id', type=str,  default='0', help='GPU used to train the network')
    
    parser.add_argument('--save', type=str,  default='./checkpoints/ModelNe40/vis_codes/', help='path to save the final model')

    parser.add_argument('--dataset', type=str, default='snRNA_snATAC', metavar='dataset',choices=['CITE_ASAP', 'snRNA_snATAC','snRNA_snmC'])

    parser.add_argument('--n_labeled', type=int, default=200, metavar='n_labeled', help='number of labeled data in the dataset')

    parser.add_argument('--pretrain_epochs', type=int, default=200, metavar='N')
    
    parser.add_argument('--batch_size', type=int, default=50, metavar='batch_size',  help='Size of batch)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of episode to train 100')

    parser.add_argument('--dropout', type=str, default=0.4, metavar='dropout', help='dropout')
    #optimizer
    parser.add_argument('--lr_rna', type=float, default=1e-4, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    parser.add_argument('--lr_atac', type=float, default=1e-4, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    
    parser.add_argument('--lr_center', type=float, default=1e-4, metavar='LR', help='learning rate for center loss (default: 0.5)')
    
    parser.add_argument('--weight_center', type=float, default=1, metavar='weight_center', help='weight center (default: 1.0)')
        
    parser.add_argument('--lr_step', type=int,  default=50,  help='how many iterations to decrease the learning rate')
    
    parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='weight_decay',  help='learning rate (default: 1e-3)')

    parser.add_argument('--per_print', type=int,  default=100, help='how many iterations to print the loss and accuracy')
                        
    args = parser.parse_args()
    print('save:', args.save)
    print('dataset:', args.dataset)
    print('n_labeled:', args.n_labeled)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.enabled = True
    ans = training(args)
    print('Best:', ans)
    # torch.backends.cudnn.enabled = False
    # if not os.path.exists(args.save):
    #     os.mkdir(args.save)

    # extract(args)
    # res = eval_func(1)