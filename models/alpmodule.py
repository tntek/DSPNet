"""
ALPModule
"""
import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt
# for unit test from spatial_similarity_module import NONLocalBlock2D, LayerNorm

class MultiProtoAsConv(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode = 'bilinear'):
        """
        ALPModule
        Args:
            proto_grid:     Grid size when doing multi-prototyping. For a 32-by-32 feature map, a size of 16-by-16 leads to a pooling window of 2-by-2
            feature_hw:     Spatial size of input feature map

        """
        super(MultiProtoAsConv, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        self.get_wight()
        kernel_size = [ ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)  ]
        self.avg_pool_op = nn.AvgPool2d( kernel_size ) # kernel_size
        self.a = 0.2 # α  $---set1:[ABD: α = 0.3] // [CMR: α = 0.2]  ---$   $---set2:[ABD: α = 0.2] ---$
       
    def get_wight(self):
        # """
        #  ------------(w1, w2, w3)---------------
        # """
        self.wight1 = 0.3  # w1
        self.wight2 = 0.8  # w2
        self.wight3 = 0.3  # w3
        N=256
        cal = torch.eye(N).float().cuda()
        for i in range(N-2):
            cal[i+1][i] = self.wight1
            cal[i+1][i+1] = self.wight2
            cal[i+1][i+2] = self.wight3
        self.cal = nn.Parameter(cal)

        
    def forward(self, mol,qry, sup_x, sup_y,s_init_seed, mode, thresh, isval = False, val_wsize = None, vis_sim = False, **kwargs):
        """
        Now supports
        Args:
            mode: 'mask'/ 'grid'. if mask, works as original prototyping
            qry: [way(1), nc, h, w]
            sup_x: [nb, nc, h, w]
            sup_y: [nb, 1, h, w]
            vis_sim: visualize raw similarities or not
        New
            mode:       'mask'/ 'grid'. if mask, works as original prototyping
            qry:        [way(1), nb(1), nc, h, w]
            sup_x:      [way(1), shot, nb(1), nc, h, w]
            sup_y:      [way(1), shot, nb(1), h, w]
            vis_sim:    visualize raw similarities or not
        """

        qry = qry.squeeze(1) # [way(1), nb(1), nc, hw] -> [way(1), nc, h, w]
        sup_x = sup_x.squeeze(0).squeeze(1) # [nshot, nc, h, w]
        sup_y = sup_y.squeeze(0) # [nshot, 1, h, w]

        def safe_norm(x, p = 2, dim = 1, eps = 1e-4):
            x_norm = torch.norm(x, p = p, dim = dim) # .detach()
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x

        if mode == 'mask': # class-level prototype only
            sup_nshot = sup_x.shape[0]
            
            out_su = self.attention(sup_x,qry)
            s_seed_ = s_init_seed[0, :, :] 
            num_sp = max(len(torch.nonzero(s_seed_[:, 0])), len(torch.nonzero(s_seed_[:, 1])))
            if (num_sp == 0):
                proto = torch.sum(out_su * sup_y, dim=(-1, -2)) \
                         / (sup_y.sum(dim=(-1, -2)) + 1e-5) # nb x C 
                cos_sim_map_sup = F.conv2d(out_su,
                                            proto[..., None, None].repeat(1, 1, 1, 1))  
                cos_sim_map_sup_t = cos_sim_map_sup.view(out_su.size()[0], 1, -1) 
                attention = cos_sim_map_sup_t.softmax(dim=-1)
                sp_center_t = proto.t().unsqueeze(0) 
                out = torch.bmm(sp_center_t, attention).view(1, sup_x.size()[1], sup_x.size()[-2], sup_x.size()[-1]) 
                out1 = out + sup_x

                proto = torch.sum(out1 * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5) 
            else: 
                if mol == 'alignLoss':
                    proto = torch.sum(out_su * sup_y, dim=(-1, -2)) \
                         / (sup_y.sum(dim=(-1, -2)) + 1e-5) 
                    cos_sim_map_sup = F.conv2d(out_su,
                                                proto[..., None, None].repeat(1, 1, 1, 1))
                    cos_sim_map_sup_t = cos_sim_map_sup.view(out_su.size()[0], 1, -1) 
                    attention = cos_sim_map_sup_t.softmax(dim=-1)
                    sp_center_t = proto.t().unsqueeze(0)
                    out = torch.bmm(sp_center_t, attention).view(1, sup_x.size()[1], sup_x.size()[-2], sup_x.size()[-1])
                    out1 = out + sup_x

                    proto = torch.sum(out1 * sup_y, dim=(-1, -2)) \
                    / (sup_y.sum(dim=(-1, -2)) + 1e-5) 
                else:
                    sp_center_list = []
                    sup_nshot = sup_x.shape[0]
                    for sup_nshot in range(sup_nshot):
                        with torch.no_grad():
                            s_seed_ = s_seed_[:num_sp, :]  # num_sp x 2
                            sp_init_center = sup_x[sup_nshot][:, s_seed_[:, 0], s_seed_[:, 1]]  
                            sp_init_center = torch.cat([sp_init_center, s_seed_.transpose(1, 0).float()], dim=0)  
                            sp_center = self.sp_center_iter(sup_x[sup_nshot], sup_y[sup_nshot], sp_init_center, n_iter=10)  
                            sp_center_list.append(sp_center) 
                        y1 = sp_center_list[0].shape[1]
                        sp_center = torch.cat(sp_center_list)
                        cos_sim_map_sup = F.conv2d(out_su,
                                                sp_center[..., None, None].repeat(1, 1, 1, 1).permute(1, 0, 2, 3)) 
                        cos_sim_map_sup_t = cos_sim_map_sup.view(sup_x.size()[0], y1, -1)
                        attention = cos_sim_map_sup_t.softmax(dim=-1)
                        sp_center_t = sp_center.unsqueeze(0) 
                        out = torch.bmm(sp_center_t, attention).view(1, sup_x.size()[1], sup_x.size()[-2], sup_x.size()[-1]) 
                        out1 = out + sup_x
                        proto = torch.sum(out1 * sup_y, dim=(-1, -2)) \
                                                            / (sup_y.sum(dim=(-1, -2)) + 1e-5)

            # proto = proto.mean(dim = 0, keepdim = True) # 1 X C, take the mean of everything
            pred_mask = F.cosine_similarity(qry, proto[..., None, None], dim=1, eps = 1e-4) * 20.0 # [1, h, w]

            vis_dict = {'proto_assign': None} # things to visualize
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict  # just a placeholder. pred_mask returned as [1, way(1), h, w]

        # no need to merge with gridconv+
        elif mode == 'gridconv':   # using local prototypes only

            input_size = qry.shape # torch.Size([1, 256, 32, 32])
            nch = input_size[1]    # 256
            out_su = self.attention(sup_x,qry)
            sup_nshot = sup_x.shape[0] # torch.Size([1, 256, 32, 32])
            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x )
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1) 
            n_sup_x = n_sup_x.permute(0, 2, 1).unsqueeze(0) 

            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)
            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0) 
            
            n_sup_x = n_sup_x.permute(0, 1, 3, 2).squeeze(0).squeeze(0)
            
            w1 = torch.mm(n_sup_x.float(),n_sup_x.permute(1, 0).float()) # 256 256
            softmax_matrix = F.softmax(w1,dim=1)
            mask_w = self.wts_near(n_sup_x, 1, 1, 1)
            add_res_w2 = mask_w*softmax_matrix.float()
            #softmax_matrix = F.softmax(add_res_w2,dim=1)
            A = 1+0.2*add_res_w2
            w_3=A*self.cal
            add_res_new = torch.mm(w_3,n_sup_x.float())

            n_sup_x = add_res_new
                     
            n_sup_x = n_sup_x.permute(1, 0).unsqueeze(0).unsqueeze(0)  

            protos = n_sup_x[sup_y_g > thresh, :] # npro, nc 
           
            pro_n = safe_norm(protos) # 56 256
            qry_n = safe_norm(qry)  # 1 256 32 32

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20 

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
            debug_assign = dists.argmax(dim = 1).float().detach()


            vis_dict = {'proto_assign': debug_assign} # things to visualize 

            if vis_sim: # return the similarity for visualization
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict


        elif mode == 'gridconv+': # local and global prototypes

            input_size = qry.shape
            nch = input_size[1]
            nb_q = input_size[0]

            sup_size = sup_x.shape[0]
            out_su = self.attention(sup_x,qry)

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  ) # 1 256 16 16

            sup_nshot = sup_x.shape[0]

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0) # 1 1 64 256
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0) # 1 1 64 256
            
            n_sup_x = n_sup_x.permute(0, 1, 3, 2).squeeze(0).squeeze(0)
           
            w1 = torch.mm(n_sup_x.float(),n_sup_x.permute(1, 0).float()) # 256 256
            softmax_matrix = F.softmax(w1,dim=1)
            mask_w = self.wts_near(n_sup_x, 1, 1, 1)
            add_res_w2 = mask_w*softmax_matrix.float()
            A = 1+self.a*add_res_w2
            w_3=A*self.cal
            add_res_new = torch.mm(w_3,n_sup_x.float())
            n_sup_x = add_res_new
            n_sup_x = n_sup_x.permute(1, 0).unsqueeze(0).unsqueeze(0)  
            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)

            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :]
 
            s_seed_ = s_init_seed[0, :, :] 
            num_sp = max(len(torch.nonzero(s_seed_[:, 0])), len(torch.nonzero(s_seed_[:, 1])))
            if (num_sp == 0):
                proto = torch.sum(out_su * sup_y, dim=(-1, -2)) \
                         / (sup_y.sum(dim=(-1, -2)) + 1e-5) # nb x C 
                cos_sim_map_sup = F.conv2d(out_su,
                                            proto[..., None, None].repeat(1, 1, 1, 1))
                cos_sim_map_sup_t = cos_sim_map_sup.view(out_su.size()[0], 1, -1)
                attention = cos_sim_map_sup_t.softmax(dim=-1)
                sp_center_t = proto.t().unsqueeze(0) 
                out = torch.bmm(sp_center_t, attention).view(1, sup_x.size()[1], sup_x.size()[-2], sup_x.size()[-1])
                out1 = out + sup_x

                proto = torch.sum(out1 * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5) # nb x C
            else: 
                if mol == 'alignLoss':
                    proto = torch.sum(out_su * sup_y, dim=(-1, -2)) \
                         / (sup_y.sum(dim=(-1, -2)) + 1e-5) # nb x C 
                    cos_sim_map_sup = F.conv2d(out_su,
                                                proto[..., None, None].repeat(1, 1, 1, 1)) 
                    cos_sim_map_sup_t = cos_sim_map_sup.view(out_su.size()[0], 1, -1)
                    attention = cos_sim_map_sup_t.softmax(dim=-1)
                    sp_center_t = proto.t().unsqueeze(0) 
                    out = torch.bmm(sp_center_t, attention).view(1, sup_x.size()[1], sup_x.size()[-2], sup_x.size()[-1])
                    out1 = out + sup_x

                    proto = torch.sum(out1 * sup_y, dim=(-1, -2)) \
                    / (sup_y.sum(dim=(-1, -2)) + 1e-5) # nb x C
                else:
                    sp_center_list = []
                    sup_nshot = sup_x.shape[0]
                    for sup_nshot in range(sup_nshot):
                        with torch.no_grad():
                            s_seed_ = s_seed_[:num_sp, :]  # num_sp x 2
                            sp_init_center = sup_x[sup_nshot][:, s_seed_[:, 0], s_seed_[:, 1]]  
                            sp_init_center = torch.cat([sp_init_center, s_seed_.transpose(1, 0).float()], dim=0)
                            sp_center = self.sp_center_iter(sup_x[sup_nshot], sup_y[sup_nshot], sp_init_center, n_iter=10)
                            sp_center_list.append(sp_center) 
                        y1 = sp_center_list[0].shape[1]
                        sp_center = torch.cat(sp_center_list)
                        cos_sim_map_sup = F.conv2d(out_su,
                                                sp_center[..., None, None].repeat(1, 1, 1, 1).permute(1, 0, 2, 3))
                        cos_sim_map_sup_t = cos_sim_map_sup.view(sup_x.size()[0], y1, -1)
                        attention = cos_sim_map_sup_t.softmax(dim=-1)
                        sp_center_t = sp_center.unsqueeze(0)
                        out = torch.bmm(sp_center_t, attention).view(1, sup_x.size()[1], sup_x.size()[-2], sup_x.size()[-1])
                        out1 = out + sup_x
                        proto = torch.sum(out1 * sup_y, dim=(-1, -2)) \
                                                            / (sup_y.sum(dim=(-1, -2)) + 1e-5)

            pro_n = safe_norm( torch.cat( [protos, proto], dim = 0 ) )

            qry_n = safe_norm(qry)

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
            raw_local_sims = dists.detach()


            debug_assign = dists.argmax(dim = 1).float()

            vis_dict = {'proto_assign': debug_assign}
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict
        
        elif mode == 'mask++': # class-level prototype only
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5) # nb x C

            proto = proto.mean(dim = 0, keepdim = True) # 1 X C, take the mean of everything
            pred_mask = F.cosine_similarity(qry, proto[..., None, None], dim=1, eps = 1e-4) * 20.0 # [1, h, w]

            vis_dict = {'proto_assign': None} # things to visualize
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict  # just a placeholder. pred_mask returned as [1, way(1), h, w]
        
        elif mode == 'bg': # using local prototypes only

            input_size = qry.shape
            nch = input_size[1]

            sup_nshot = sup_x.shape[0]

            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op( sup_x  )

            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0) # way(1),nb, hw, nc
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)

            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0)

            protos = n_sup_x[sup_y_g > thresh, :] # npro, nc
            pro_n = safe_norm(protos)
            qry_n = safe_norm(qry)
            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20

            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
            debug_assign = dists.argmax(dim = 1).float().detach()

            vis_dict = {'proto_assign': debug_assign} # things to visualize

            if vis_sim: # return the similarity for visualization
                vis_dict['raw_local_sims'] = dists.clone().detach()

            return pred_grid, [debug_assign], vis_dict

        else:
            raise NotImplementedError

    
    def avg_near(self, tmp):
        N = tmp.shape[0]
        cal = torch.eye(N).cuda()
        for i in range(N-2):
            cal[i+1][i] = 1
            cal[i+1][i+1] = 1
            cal[i+1][i+2] = 1
    
        add_res = torch.mm(cal.float(),tmp.float())

        diag = torch.full([N],1.0/3.0)
        cal = torch.diag_embed(diag).cuda()
        cal[0][0] = 1
        cal[N-1][N-1] = 1
        res = torch.mm(cal.float(),add_res)
        return res

    def wts_near(self, tmp, weights_1, weight_2, weight_3):

        N = tmp.shape[0]

        cal = torch.eye(N).float().cuda()
        for i in range(N-2):
            cal[i+1][i] = weights_1
            cal[i+1][i+1] = weight_2
            cal[i+1][i+2] = weight_3

        return cal
    
    def  attention(self,sup_x,qry):
        reduce_dim = 256
        #key_conv = nn.Conv2d(in_channels=reduce_dim, out_channels=reduce_dim, kernel_size=1).cuda()
        #qu_conv = nn.Conv2d(in_channels=reduce_dim, out_channels=reduce_dim, kernel_size=1).cuda()
        #v_conv = nn.Conv2d(in_channels=reduce_dim, out_channels=reduce_dim, kernel_size=1).cuda()
        x_sup = sup_x.view(sup_x.size()[0],sup_x.size()[1], -1)
        x_que =qry.view(qry.size()[0], qry.size()[1], -1)
        x_sup_g = sup_x.view(sup_x.size()[0], sup_x.size()[1], -1) 

        x_que_norm = torch.norm(x_que, p=2, dim=1, keepdim=True) 
        x_sup_norm = torch.norm(x_sup, p=2, dim=1, keepdim=True) 

        x_sup_norm = x_sup_norm.permute(0, 2, 1)
        x_qs_norm = torch.matmul( x_sup_norm, x_que_norm)
        x_sup = x_sup.permute(0, 2, 1)
        x_qs = torch.matmul(x_sup, x_que)
        x_qs = x_qs / (x_qs_norm + 1e-5)
        R_qs = x_qs
        attention = R_qs.softmax(dim=-1)

        proj_value = x_sup_g 

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(1, sup_x.size()[1], sup_x.size()[2], sup_x.size()[3])
        out = out + sup_x
        out_su = out # sup_x[[sup_nshot]]
        return out_su
    
    def sp_center_iter(self, supp_feat, supp_mask, sp_init_center, n_iter):
        c_xy, num_sp = sp_init_center.size()
        _, h, w = supp_feat.size()
        h_coords = torch.arange(h).view(h, 1).contiguous().repeat(1, w).unsqueeze(0).float().cuda()
        w_coords = torch.arange(w).repeat(h, 1).unsqueeze(0).float().cuda() 
        supp_feat = torch.cat([supp_feat, h_coords, w_coords], 0)
        supp_feat_roi = supp_feat[:, (supp_mask == 1).squeeze()] 

        num_roi = supp_feat_roi.size(1) 
        supp_feat_roi_rep = supp_feat_roi.unsqueeze(-1).repeat(1, 1, num_sp)
        sp_center = torch.zeros_like(sp_init_center).cuda()  # (C + xy) x num_sp

        for i in range(n_iter):
            # Compute association between each pixel in RoI and superpixel
            if i == 0:
                sp_center_rep = sp_init_center.unsqueeze(1).repeat(1, num_roi, 1)
            else:
                sp_center_rep = sp_center.unsqueeze(1).repeat(1, num_roi, 1)
            assert supp_feat_roi_rep.shape == sp_center_rep.shape  # (C + xy) x num_roi x num_sp
            dist = torch.pow(supp_feat_roi_rep - sp_center_rep, 2.0) 
            feat_dist = dist[:-2, :, :].sum(0)
            spat_dist = dist[-2:, :, :].sum(0)
            total_dist = torch.pow(feat_dist/100 + spat_dist / 100, 0.5)
            p2sp_assoc = torch.neg(total_dist).exp()
            p2sp_assoc = p2sp_assoc / (p2sp_assoc.sum(0, keepdim=True))  # num_roi x num_sp

            sp_center = supp_feat_roi_rep * p2sp_assoc.unsqueeze(0)  # (C + xy) x num_roi x num_sp
            sp_center = sp_center.sum(1)

        return sp_center[:-2, :]

