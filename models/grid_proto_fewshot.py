"""
ALPNet
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alpmodule import MultiProtoAsConv
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder
from util.seed_init import place_seed_points_d
# DEBUG
from pdb import set_trace

import pickle
import torchvision

# options for type of prototypes
FG_PROT_MODE = 'gridconv+'
BG_PROT_MODE = 'gridconv' 
# thresholds for deciding class of prototypes
FG_THRESH = 0.95
BG_THRESH = 0.95



class FewShotSeg(nn.Module):
    """
    ALPNet
    Args:
        in_channels:        Number of input channels
        cfg:                Model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None):
        super(FewShotSeg, self).__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.get_encoder(in_channels)
        self.get_cls()

    def get_encoder(self, in_channels):
        # if self.config['which_model'] == 'deeplab_res101':
        if self.config['which_model'] == 'dlfcn_res101':
            use_coco_init = self.config['use_coco_init']
            self.encoder = TVDeeplabRes101Encoder(use_coco_init)

        else:
            raise NotImplementedError(f'Backbone network {self.config["which_model"]} not implemented')

        if self.pretrained_path:
            check = torch.load(self.pretrained_path)
            self.load_state_dict(torch.load(self.pretrained_path),False)
            print(f'###### Pre-trained model f{self.pretrained_path} has been loaded ######')


    def get_cls(self):
        """
        Obtain the similarity-based classifier
        """
        proto_hw = self.config["proto_grid_size"]
        feature_hw = self.config["feature_hw"]
        assert self.config['cls_name'] == 'grid_proto'
        if self.config['cls_name'] == 'grid_proto':
            self.cls_unit = MultiProtoAsConv(proto_grid = [proto_hw, proto_hw], feature_hw =  self.config["feature_hw"]) # when treating it as ordinary prototype
        else:
            raise NotImplementedError(f'Classifier {self.config["cls_name"]} not implemented')


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, isval, val_wsize, show_viz = False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
            show_viz: return the visualization dictionary
        """
        mol = 'Loss'
        # ('Please go through this piece of code carefully')
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs) 

        # print("aa", n_ways, "bb", n_shots, "cc", n_queries)

        assert n_ways == 1, "Multi-shot has not been implemented yet" # NOTE: actual shot in support goes in batch dimension
        assert n_queries == 1

        sup_bsize = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        qry_bsize = qry_imgs[0].shape[0]

        assert sup_bsize == qry_bsize == 1

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)

        img_fts = self.encoder(imgs_concat, low_level = False)
        fts_size = img_fts.shape[-2:] 

        # print("aa", img_fts.shape)

        supp_fts = img_fts[:n_ways * n_shots * sup_bsize].view(
            n_ways, n_shots, sup_bsize, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * sup_bsize:].view(
            n_queries, qry_bsize, -1, *fts_size)        # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        fore_mask = torch.autograd.Variable(fore_mask, requires_grad = True)
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H' x W'
        s_y = fore_mask[0][0][0]

        init_seed_list = []
        mask = (s_y == 1).float()  # H x W 

        init_seed = place_seed_points_d(mask, down_stride=8, max_num_sp=5,
                                                            avg_sp_area=100)
        init_seed_list.append(init_seed.unsqueeze(0))
        s_init_seed = torch.cat(init_seed_list).cuda()

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        visualizes = [] # the buffer for visualization

        for epi in range(1):
            fg_masks = []    # keep the way part
            res_fg_msk = torch.stack([F.interpolate(fore_mask_w, size = fts_size, mode = 'bilinear') for fore_mask_w in fore_mask], dim = 0) # [nway, ns, nb, nh', nw']
            res_bg_msk = torch.stack([F.interpolate(back_mask_w, size = fts_size, mode = 'bilinear') for back_mask_w in back_mask], dim = 0) # [nway, ns, nb, nh', nw']


            scores      = []
            assign_maps = []
            bg_sim_maps = []
            fg_sim_maps = []

            _raw_score, _, aux_attr = self.cls_unit(mol,qry_fts, supp_fts, res_bg_msk,s_init_seed, mode = BG_PROT_MODE, thresh = BG_THRESH, isval = isval, val_wsize = val_wsize, vis_sim = show_viz  )
            scores.append(_raw_score)
            assign_maps.append(aux_attr['proto_assign'])
            if show_viz:
                bg_sim_maps.append(aux_attr['raw_local_sims'])

            for way, _msk in enumerate(res_fg_msk): #'mask++'
                _raw_score, _, aux_attr = self.cls_unit(mol,qry_fts, supp_fts, _msk.unsqueeze(0),s_init_seed, mode = FG_PROT_MODE if F.avg_pool2d(_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', thresh = FG_THRESH, isval = isval, val_wsize = val_wsize, vis_sim = show_viz  )

                scores.append(_raw_score)
                if show_viz:
                    fg_sim_maps.append(aux_attr['raw_local_sims'])

            pred = torch.cat(scores, dim=1)   # N x (1 + Wa) x H' x W'

            pred_int = F.interpolate(pred, size=img_size, mode='bilinear')

            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred, pred_int, supp_fts[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi
             

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        assign_maps = torch.stack(assign_maps, dim = 1)
        bg_sim_maps = torch.stack(bg_sim_maps, dim = 1) if show_viz else None
        fg_sim_maps = torch.stack(fg_sim_maps, dim = 1) if show_viz else None

        return output, align_loss / sup_bsize, [bg_sim_maps, fg_sim_maps], assign_maps


    # Batch was at the outer loop
    def alignLoss(self, qry_fts, pred, pred_int,supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x H x W
            supp_fts: embedding fatures for support images
                expect shape: Wa x Sh x C x H' x W'
            fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        mol = 'alignLoss'
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1).unsqueeze(0)  #1 x  N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        
        pred_mask_int = pred_int.argmax(dim=1).unsqueeze(0)  # N x 1 x H' x W'
        binary_masks_int = [pred_mask_int == i for i in range(1 + n_ways)]
        pred_mask_int = binary_masks_int[1].float()
        s_y = pred_mask_int

        init_seed_list = []
        mask = (s_y[0,:,:] == 1).float()  # H x W  473 473 [0.0, 1.0]

        init_seed = place_seed_points_d(mask, down_stride=8, max_num_sp=5,
                                                            avg_sp_area=100)
        init_seed_list.append(init_seed.unsqueeze(0))
        s_init_seed = torch.cat(init_seed_list)

        # skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        # FIXME: fix this in future we here make a stronger assumption that a positive class must be there to avoid undersegmentation/ lazyness
        skip_ways = []

        qry_fts = qry_fts.unsqueeze(0).unsqueeze(2) 

        ### end of added part

        loss = []
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way: way + 1, shot: shot + 1] # actual local query [way(1), nb(1, nb is now nshot), nc, h, w]

                qry_pred_fg_msk = F.interpolate(binary_masks[way + 1].float(), size = img_fts.shape[-2:], mode = 'bilinear') # [1 (way), n (shot), h, w]

                # background
                qry_pred_bg_msk = F.interpolate(binary_masks[0].float(), size = img_fts.shape[-2:], mode = 'bilinear') # 1, n, h ,w
                scores = []

                _raw_score_bg, _, _ = self.cls_unit(mol=mol,qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_bg_msk.unsqueeze(-3),s_init_seed=s_init_seed, mode = BG_PROT_MODE, thresh = BG_THRESH )

                scores.append(_raw_score_bg)

                _raw_score_fg, _, _ = self.cls_unit(mol=mol,qry = img_fts, sup_x = qry_fts, sup_y = qry_pred_fg_msk.unsqueeze(-3),s_init_seed=s_init_seed, mode = FG_PROT_MODE if F.avg_pool2d(qry_pred_fg_msk, 4).max() >= FG_THRESH and FG_PROT_MODE != 'mask' else 'mask', thresh = FG_THRESH )

                scores.append(_raw_score_fg)

                supp_pred = torch.cat(scores, dim=1)  # N x (1 + Wa) x H' x W'
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:], mode='bilinear')

                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss.append( F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways)

        return torch.sum( torch.stack(loss))
