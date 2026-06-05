import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models import builder
from mmdet.models import DETECTORS
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.ops import Voxelization, DynamicScatter
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head import BEVFormerHead
from projects.mmdet3d_plugin.mask2map.deformable_detr.models.ops.modules.ms_deform_attn import MSDeformAttn
from mmdet3d.core import bbox3d2result
from projects.mmdet3d_plugin.bevformer.modules.decoder import inverse_sigmoid
from projects.mmdet3d_plugin.mask2map.dense_heads.mask2map_head_1p import Mask2MapHead_1Phase

def get_runner():
    from projects.mmdet3d_plugin.bevformer.apis.mmdet_train import runner 
    return runner 


@DETECTORS.register_module()
class Mask2Map(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        obj_pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        obj_train_cfg=None,
        obj_test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        modality="vision",
        lidar_encoder=None,
    ):

        super().__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            obj_pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            obj_train_cfg,
            obj_test_cfg,
            pretrained,
        )
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
        self.modality = modality
        if self.modality == "fusion" and lidar_encoder is not None:
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {"voxelize": voxelize_module, "backbone": builder.build_middle_encoder(lidar_encoder["backbone"])}
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)
            
        self.deformable_latent_fusion = MSDeformAttn(d_model=256, n_heads=4, n_levels=2, n_points=16) 


        #For reference points that lie outside the range of [0, 1], the sampled values will be zero 
        #reference points from OD to map prediction: 
        h = w = obj_pts_bbox_head['bev_h']
        h_metric = w_metric = 51.2*2
        metric_positions = torch.zeros((h, w, 2))
        for h_indx in range(h): 
            metric_positions[h_indx, :, 1] += h_metric*(float(h_indx)/float(h) + 0.5/float(h)) - h_metric/2.
        for w_indx in range(w):
            metric_positions[:, w_indx, 0] += w_metric*(float(w_indx)/float(w) + 0.5/float(w)) - w_metric/2.
        reference_points_obj_to_map = torch.zeros(h, w, 2)
        for h_indx in range(h): 
            reference_points_obj_to_map[h_indx, :, 1] = metric_positions[h_indx, :, 1]/30.
        for w_indx in range(w): 
            reference_points_obj_to_map[:, w_indx, 0] = metric_positions[:, w_indx, 0]/15.
        reference_points_obj_to_map /= 2
        reference_points_obj_to_map[:, :, 0] += 0.5
        reference_points_obj_to_map[:, :, 1] += 0.5
        reference_points_obj_to_map = torch.flatten(reference_points_obj_to_map, start_dim=0, end_dim=1) 

        #reference points map prediction to OD: 
        h = 200 
        w = 100
        h_metric = 60
        w_metric = 30
        metric_positions = torch.zeros((h, w, 2))
        for h_indx in range(h): 
            metric_positions[h_indx, :, 1] += h_metric*(float(h_indx)/float(h) + 0.5/float(h)) - h_metric/2.
        for w_indx in range(w):
            metric_positions[:, w_indx, 0] += w_metric*(float(w_indx)/float(w) + 0.5/float(w)) - w_metric/2.
        reference_points_map_to_obj = torch.zeros(h, w, 2)
        for h_indx in range(h): 
            reference_points_map_to_obj[h_indx, :, 1] = metric_positions[h_indx, :, 1]/51.2
        for w_indx in range(w): 
            reference_points_map_to_obj[:, w_indx, 0] = metric_positions[:, w_indx, 0]/51.2
        reference_points_map_to_obj /= 2
        reference_points_map_to_obj[:, :, 0] += 0.5
        reference_points_map_to_obj[:, :, 1] += 0.5
        reference_points_map_to_obj = torch.flatten(reference_points_map_to_obj, start_dim=0, end_dim=1)

        #reference points from map to map: 
        h = 200
        w = 100
        reference_points_map_to_map = torch.zeros((h, w, 2))
        for h_indx in range(h): 
            reference_points_map_to_map[h_indx, :, 1] += float(h_indx)/float(h) + 0.5/float(h)
        for w_indx in range(w):
            reference_points_map_to_map[:, w_indx, 0] += float(w_indx)/float(w) + 0.5/float(w)
        reference_points_map_to_map = torch.flatten(reference_points_map_to_map, start_dim=0, end_dim=1)


        #reference points from OD to OD: 
        h = w = obj_pts_bbox_head['bev_h']
        reference_points_obj_to_obj = torch.zeros((h, w, 2))
        for h_indx in range(h): 
            reference_points_obj_to_obj[h_indx, :, 1] += float(h_indx)/float(h) + 0.5/float(h)
        for w_indx in range(w):
            reference_points_obj_to_obj[:, w_indx, 0] += float(w_indx)/float(w) + 0.5/float(w)
        reference_points_obj_to_obj = torch.flatten(reference_points_obj_to_obj, start_dim=0, end_dim=1)

        reference_points_all_to_obj = torch.cat((reference_points_obj_to_obj, reference_points_map_to_obj), dim=0)
        reference_points_all_to_map = torch.cat((reference_points_obj_to_map, reference_points_map_to_map), dim=0)
        self.reference_points = torch.stack((reference_points_all_to_obj, reference_points_all_to_map), dim=1)
        self.reference_points = torch.unsqueeze(self.reference_points, 0).cuda()

        # h = 200
        # w = 100
        # reference_points = torch.zeros((h, w, 2))
        # for h_indx in range(h): 
        #     reference_points[h_indx, :, 1] += float(h_indx)/float(h) + 0.5/float(h)
        # for w_indx in range(w):
        #     reference_points[:, w_indx, 0] += float(w_indx)/float(w) + 0.5/float(w)
        # self.reference_points = torch.flatten(reference_points, start_dim=0, end_dim=1) 
        # self.reference_points = torch.cat((self.reference_points, self.reference_points), dim=0)
        # self.reference_points = torch.stack((self.reference_points, self.reference_points), dim=1)
        # self.reference_points = torch.unsqueeze(self.reference_points, 0).cuda()

        self.cross_attns_map_to_obj = torch.nn.ModuleList()
        for _ in range(6): 
            self.cross_attns_map_to_obj.append(torch.nn.MultiheadAttention(embed_dim=256, num_heads=4))

        self.ffns_map_to_obj = torch.nn.ModuleList()
        for _ in range(6):
            ffn = torch.nn.ModuleList()
            ffn.append(torch.nn.Linear(256, 512))
            ffn.append(torch.nn.ReLU())
            ffn.append(torch.nn.Dropout(p=0.1))
            ffn.append(torch.nn.Linear(512, 256))
            ffn.append(torch.nn.Dropout(p=0.1))
            self.ffns_map_to_obj.append(ffn)

        self.norms_map_to_obj = torch.nn.ModuleList()
        for _ in range(6):
            self.norms_map_to_obj.append(torch.nn.LayerNorm(256))

        
        # self.cross_attns_obj_to_map = torch.nn.ModuleList()
        # for _ in range(6): 
        #     self.cross_attns_obj_to_map.append(torch.nn.MultiheadAttention(embed_dim=256, num_heads=4))

        # self.ffns_obj_to_map = torch.nn.ModuleList()
        # for _ in range(6):
        #     ffn = torch.nn.ModuleList()
        #     ffn.append(torch.nn.Linear(256, 512))
        #     ffn.append(torch.nn.ReLU())
        #     ffn.append(torch.nn.Dropout(p=0.1))
        #     ffn.append(torch.nn.Linear(512, 256))
        #     ffn.append(torch.nn.Dropout(p=0.1))
        #     self.ffns_obj_to_map.append(ffn)

        # self.norms_obj_to_map = torch.nn.ModuleList()
        # for _ in range(6):
        #     self.norms_obj_to_map.append(torch.nn.LayerNorm(256))

        self.num_iterations_done = 0

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    #@auto_fp16(apply_to=("img"), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(
        self,
        pts_feats,
        lidar_feat,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
        prev_bev=None,
        gt_depth=None,
        gt_seg_mask=None,
        gt_pv_seg_mask=None,
        obj_gt_bboxes_3d=None,
        obj_gt_labels_3d=None, 
        obj_gt_names=None
    ):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        #determine the factors to use for the inter-task attention outputs and the residual connections aruond the attention blocks 
        runner = get_runner()
        current_epoch = runner.epoch  #int(self.num_iterations_done/7033) #only correct for BS of 4 
        current_epoch = min(current_epoch, 3) 
        attention_factor = (current_epoch/11)*0.5
        residual_factor = 1-attention_factor

        if isinstance(self.pts_bbox_head, Mask2MapHead_1Phase):
            #only use attention across tasks during phase 2
            attention_factor = 0.
            residual_factor = 1.

        #Get the encoder features from both heads: 
        #BEVFormer head
        obj_encoder_feats = self.obj_pts_bbox_head(pts_feats, img_metas, prev_bev['obj'], only_bev=True)
        residual_obj = obj_encoder_feats

        #Mask2Map head
        map_encoder_feats = self.pts_bbox_head(pts_feats, lidar_feat, img_metas, prev_bev['map'], only_bev=True)
        residual_map = map_encoder_feats['bev']

        #ITSA between the two tasks' encoder features
        bs = obj_encoder_feats.shape[0]
        reference_points = self.reference_points
        for i in range(bs-1): 
            reference_points = torch.cat((reference_points, self.reference_points), dim=0)
        refined_feats = self.deformable_latent_fusion(query = torch.cat((obj_encoder_feats, map_encoder_feats['bev']), dim=1), 
                                                      input_flatten = torch.cat((obj_encoder_feats, map_encoder_feats['bev']), dim=1), 
                                                      input_spatial_shapes = torch.tensor([[self.obj_pts_bbox_head.bev_h, self.obj_pts_bbox_head.bev_w], [200, 100]]).cuda(),
                                                      input_level_start_index = torch.tensor([0,self.obj_pts_bbox_head.bev_h*self.obj_pts_bbox_head.bev_w]).cuda(),
                                                      reference_points = reference_points)
        
        obj_encoder_feats = residual_obj
        map_encoder_feats['bev'] = attention_factor*refined_feats[:, self.obj_pts_bbox_head.bev_h*self.obj_pts_bbox_head.bev_w:, ...] + residual_factor*residual_map

        #test code:------- 
        test = False
        if test: 
            obj_test = torch.reshape(obj_encoder_feats, (1, 150, 150, 256))
            map_test = torch.reshape(map_encoder_feats['bev'], (1, 200, 100, 256))
            residual_obj_test = torch.reshape(residual_obj, (1, 150, 150, 256))
            residual_map_test = torch.reshape(residual_map, (1, 200, 100, 256))
        
            import matplotlib.pyplot as plt 
            plt.imsave('./obj_test_residual.png', obj_test[0, :, :, 0].detach().cpu().numpy())
            plt.imsave('./map_test_residual.png', map_test[0, :, :, 0].detach().cpu().numpy())
            plt.imsave('./obj_input.png', residual_obj_test[0, :, :, 0].detach().cpu().numpy())
            plt.imsave('./map_input.png', residual_map_test[0, :, :, 0].detach().cpu().numpy())

            print(f'obj_test: max:{torch.max(obj_test)}, min:{torch.min(obj_test)}')
            print(f'map_test: max:{torch.max(map_test)}, min:{torch.min(map_test)}')
            print(f'obj_input: max:{torch.max(residual_obj_test)}, min:{torch.min(residual_obj_test)}')
            print(f'map_input: max:{torch.max(residual_map_test)}, min:{torch.min(residual_map_test)}')
        #-----------------

        if isinstance(self.pts_bbox_head, Mask2MapHead_1Phase):
            #Get the decoder outputs given the refined encoder features for both heads:
            #BEVFormer head
            outs_obj = self.obj_pts_bbox_head(pts_feats, img_metas, prev_bev, bev_embed=obj_encoder_feats)

            #Mask2Map head
            outs = self.pts_bbox_head(pts_feats, lidar_feat, img_metas, prev_bev, encoder_feats=map_encoder_feats)

            losses = dict()
            
            #Mask2Map head losses
            depth = outs.pop("depth")
            # calculate depth loss
            if gt_depth is not None:
                loss_depth = self.pts_bbox_head.transformer.encoder.get_depth_loss(gt_depth, depth)
                if digit_version(TORCH_VERSION) >= digit_version("1.8"):
                    loss_depth = torch.nan_to_num(loss_depth)
                losses.update(loss_depth=loss_depth)

            gt_vecs_list = copy.deepcopy(gt_bboxes_3d)
            device = gt_labels_3d[0].device

            #labels for all map instances
            gt_labels_list = [gt_bboxes.gt_labels.to(device) for gt_bboxes in gt_vecs_list]                                     
            #2D bboxes for all map instances that cover all points for that instance
            gt_bboxes_list = [gt_bboxes.bbox_condi.to(device) for gt_bboxes in gt_vecs_list]
            #ordered point sets for all map instances
            gt_pts_list = [gt_bboxes.fixed_num_sampled_points_condi.to(device) for gt_bboxes in gt_vecs_list]
            #probbly the noised map instances for the denoising loss 
            gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points_v2_condi.to(device) for gt_bboxes in gt_vecs_list]

            gt_segms_list = [gt_bboxes.instance_segments_condi.to(device) for gt_bboxes in gt_vecs_list]
            loss_inputs = [gt_bboxes_list, gt_labels_list, gt_pts_list, gt_seg_mask, gt_pv_seg_mask, gt_shifts_pts_list, gt_segms_list, outs]

            losses_pts = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
            losses.update(losses_pts)


            #BEVFormer head losses
            if "seg_preds" not in outs_obj.keys():
                obj_gt_labels_3d = [torch.tensor(label, device='cuda') for label in obj_gt_labels_3d]
                det_loss_inputs = [obj_gt_bboxes_3d, obj_gt_labels_3d, outs_obj]
                det_losses = self.obj_pts_bbox_head.loss(*det_loss_inputs, img_metas=img_metas)
                for k,v in det_losses.items(): 
                    losses[f'obj_{k}'] = 3.*v #scale the det losses to match the magnitude og the Mask2Map losses 
            else: 
                raise(Exception("The BEVFormer head shouuld only perform OD"))

            return losses

        #Get the decoder outputs given the refined encoder features for both heads:
        #BEVFormer head
        #-------------------------------------------------------------------------------------------------
        #outs_obj = self.obj_pts_bbox_head(pts_feats, img_metas, prev_bev, bev_embed=obj_encoder_feats)

        bs, num_cam, _, _, _ = pts_feats[0].shape
        dtype = pts_feats[0].dtype
        object_query_embeds = self.obj_pts_bbox_head.query_embedding.weight.to(dtype)
        bev_queries = self.obj_pts_bbox_head.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.obj_pts_bbox_head.bev_h, self.obj_pts_bbox_head.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.obj_pts_bbox_head.positional_encoding(bev_mask).to(dtype)

        # outputs = self.obj_pts_bbox_head.transformer(
        #     pts_feats,
        #     bev_queries,
        #     object_query_embeds,
        #     self.obj_pts_bbox_head.bev_h,
        #     self.obj_pts_bbox_head.bev_w,
        #     grid_length=(self.obj_pts_bbox_head.real_h / self.obj_pts_bbox_head.bev_h,
        #                     self.obj_pts_bbox_head.real_w / self.obj_pts_bbox_head.bev_w),
        #     bev_pos=bev_pos,
        #     reg_branches=self.obj_pts_bbox_head.reg_branches if self.obj_pts_bbox_head.with_box_refine else None,  # noqa:E501
        #     cls_branches=self.obj_pts_bbox_head.cls_branches if self.obj_pts_bbox_head.as_two_stage else None,
        #     img_metas=img_metas,
        #     prev_bev=prev_bev,
        #     bev_embed=obj_encoder_feats
        # )

        bs = pts_feats[0].size(0)
        query_pos_obj, query_obj = torch.split(
            object_query_embeds, self.obj_pts_bbox_head.transformer.embed_dims, dim=1)
        query_pos_obj = query_pos_obj.unsqueeze(0).expand(bs, -1, -1)
        query_obj = query_obj.unsqueeze(0).expand(bs, -1, -1)
        reference_points_obj = self.obj_pts_bbox_head.transformer.reference_points(query_pos_obj)
        reference_points_obj = reference_points_obj.sigmoid()
        init_reference_out_obj = reference_points_obj

        query_obj = query_obj.permute(1, 0, 2)
        query_pos_obj = query_pos_obj.permute(1, 0, 2)
        bev_embed_obj = obj_encoder_feats.permute(1, 0, 2)

        reg_branches_obj=self.obj_pts_bbox_head.reg_branches if self.obj_pts_bbox_head.with_box_refine else None  # noqa:E501
        cls_branches_obj=self.obj_pts_bbox_head.cls_branches if self.obj_pts_bbox_head.as_two_stage else None

        # inter_states, inter_references = self.obj_pts_bbox_head.transformer.decoder(
        #     query=query,
        #     key=None,
        #     value=bev_embed,
        #     query_pos=query_pos,
        #     reference_points=reference_points,
        #     reg_branches=reg_branches,
        #     cls_branches=cls_branches,
        #     spatial_shapes=torch.tensor([[self.obj_pts_bbox_head.bev_h, self.obj_pts_bbox_head.bev_w]], device=query.device),
        #     level_start_index=torch.tensor([0], device=query.device),
        #     img_metas=img_metas)

        output_obj = query_obj
        intermediate_obj = []
        intermediate_reference_points_obj = []
        kwargs_obj = dict(key=None, 
                      value=bev_embed_obj,
                      query_pos=query_pos_obj, 
                      cls_branches=cls_branches_obj, 
                      spatial_shapes=torch.tensor([[self.obj_pts_bbox_head.bev_h, self.obj_pts_bbox_head.bev_w]], device=query_obj.device),
                      level_start_index=torch.tensor([0], device=query_obj.device),
                      img_metas=img_metas
                      )
        # for lid, layer in enumerate(self.obj_pts_bbox_head.transformer.decoder.layers):

        #     reference_points_input_obj = reference_points_obj[..., :2].unsqueeze(
        #         2)  # BS NUM_QUERY NUM_LEVEL 2
        #     output_obj = layer(
        #         output_obj,
        #         reference_points=reference_points_input_obj,
        #         key_padding_mask=None,
        #         **kwargs_obj)
        #     output_obj = output_obj.permute(1, 0, 2)

        #     if reg_branches_obj is not None:
        #         tmp = reg_branches_obj[lid](output_obj)

        #         assert reference_points_obj.shape[-1] == 3

        #         new_reference_points = torch.zeros_like(reference_points_obj)
        #         new_reference_points[..., :2] = tmp[
        #             ..., :2] + inverse_sigmoid(reference_points_obj[..., :2])
        #         new_reference_points[..., 2:3] = tmp[
        #             ..., 4:5] + inverse_sigmoid(reference_points_obj[..., 2:3])

        #         new_reference_points = new_reference_points.sigmoid()

        #         reference_points_obj = new_reference_points.detach()

        #     output_obj = output_obj.permute(1, 0, 2)
        #     if self.obj_pts_bbox_head.transformer.decoder.return_intermediate:
        #         intermediate_obj.append(output_obj)
        #         intermediate_reference_points_obj.append(reference_points_obj)

        #     #TODO: for each refinement layer, the queries in output need to be 
        #     # cross-attended with the map prediction queries of the corresponding refinement layer 

        
        
        #-------------------------------------------------------------------------------------------------

        #Mask2Map head
        #TODO: this is for the pts_bbox_head for phase 2, phase 1 needs to be handled separately 
        # (does phase 1 even compute the MaskGuidedMapDecoder?, otherwise only phase 2 needs to be dealt with)
        #-------------------------------------------------------------------------------------------------
        #outs = self.pts_bbox_head(pts_feats, lidar_feat, img_metas, prev_bev, encoder_feats=map_encoder_feats)

        num_vec = self.pts_bbox_head.num_vec_one2one

        bs, num_cam, _, _, _ = pts_feats[0].shape
        dtype = pts_feats[0].dtype

        pts_query = self.pts_bbox_head.pts_embedding.weight.unsqueeze(0)

        bev_queries = None
        bev_mask = None
        bev_pos = None

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = torch.zeros([num_vec, num_vec]).bool().to(pts_feats[0].device)
        self_attn_mask[self.pts_bbox_head.num_vec_one2one :, 0 : self.pts_bbox_head.num_vec_one2one] = True
        self_attn_mask[0 : self.pts_bbox_head.num_vec_one2one, self.pts_bbox_head.num_vec_one2one :] = True

        # outputs = self.pts_bbox_head.transformer(
        #     pts_feats,
        #     lidar_feat,
        #     bev_queries,
        #     pts_query,
        #     self.pts_bbox_head.bev_h,
        #     self.pts_bbox_head.bev_w,
        #     grid_length=(self.pts_bbox_head.real_h / self.pts_bbox_head.bev_h, self.pts_bbox_head.real_w / self.pts_bbox_head.bev_w),
        #     bev_pos=bev_pos,
        #     reg_branches=(self.pts_bbox_head.reg_branches if self.pts_bbox_head.with_box_refine else None),  # noqa:E501
        #     cls_branches=self.pts_bbox_head.cls_branches if self.pts_bbox_head.as_two_stage else None,
        #     img_metas=img_metas,
        #     prev_bev=prev_bev,
        #     self_attn_mask=self_attn_mask,
        #     num_vec=num_vec,
        #     num_pts_per_vec=self.pts_bbox_head.num_pts_per_vec,
        #     encoder_feats=map_encoder_feats
        # )

        grid_length=(self.pts_bbox_head.real_h / self.pts_bbox_head.bev_h, self.pts_bbox_head.real_w / self.pts_bbox_head.bev_w)
        kwargs_map = dict(img_metas=img_metas,
                  self_attn_mask=self_attn_mask,
                  num_vec=num_vec, 
                  num_pts_per_vec=self.pts_bbox_head.num_pts_per_vec,
                  encoder_feats=map_encoder_feats)
        (mask_pred_list, cls_pred_list, mask_aware_query_feat, bev_embed_map, bev_embed_ms, depth, dn_information, kwargs_map) = self.pts_bbox_head.transformer.IMPNet(
            pts_feats, lidar_feat, bev_queries, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, grid_length=grid_length, bev_pos=bev_pos, prev_bev=prev_bev, **kwargs_map
        )

        # MMPNet
        mask_dict, known_masks, known_bid, map_known_indice = dn_information

        mask_aware_query_feat, mask_aware_query_pos = self.pts_bbox_head.transformer.PositionalQueryGenerator(
            mask_aware_query_feat,
            pts_query,
            mask_pred_list[-1],
            known_masks=known_masks,
            known_bid=known_bid,
            map_known_indice=map_known_indice,
        )

        query_map, key_map, value_map, mask_aware_query, mask_aware_query_norm = self.pts_bbox_head.transformer.GeometricFeatureExtractor(
            mask_pred_list,
            mask_aware_query_feat,
            mask_aware_query_pos,
            bev_embed_map,
        )
        #query_map is computed solely based on mask_aware_query_feat and mask_aware_query_pos, somewhat strange to have that as part of GeometricFeatureExtractor, 
        #would make more sense to compute that as part of the PositionalQueryGenerator, as shown in the paper
        #key_map and value_map are computed based on mask_pred_list

        reg_branches_map=(self.pts_bbox_head.reg_branches if self.pts_bbox_head.with_box_refine else None)  # noqa:E501
        cls_branches_map=self.pts_bbox_head.cls_branches if self.pts_bbox_head.as_two_stage else None
        
        # inter_states, inter_references_out, init_reference_out = self.pts_bbox_head.transformer.MaskGuidedMapDecoder(
        #     query,
        #     key,
        #     value,
        #     bev_embed_ms,
        #     mask_aware_query,
        #     mask_aware_query_norm,
        #     reg_branches_map,
        #     cls_branches_map,
        #     **kwargs,
        # )

        bs = mask_aware_query.shape[0]
        q = query_map.shape[1]//bs
        attn_output, _ = self.pts_bbox_head.transformer.m_attn(query_map, key_map, value_map)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = attn_output.reshape(bs, q, -1, self.pts_bbox_head.transformer.embed_dims * 2)
        attn_output = attn_output.reshape(bs, -1, self.pts_bbox_head.transformer.embed_dims * 2)

        gate = torch.sigmoid(self.pts_bbox_head.transformer.lin_ih(attn_output) + self.pts_bbox_head.transformer.lin_hh(mask_aware_query_norm))
        query_map = mask_aware_query + self.pts_bbox_head.transformer.out_proj(attn_output + gate * (self.pts_bbox_head.transformer.lin_self(mask_aware_query_norm) - attn_output))
        query_map = query_map + self.pts_bbox_head.transformer.mlp2(self.pts_bbox_head.transformer.norm2(query_map))

        query_pos_map, query_feat_map = torch.split(query_map, self.pts_bbox_head.transformer.embed_dims, dim=-1)

        reference_points_map = self.pts_bbox_head.transformer.reference_points(query_pos_map)
        reference_points_map = reference_points_map.sigmoid()
        init_reference_out_map = reference_points_map

        query_feat_map = query_feat_map.permute(1, 0, 2) 
        query_pos_map = query_pos_map.permute(1, 0, 2)
        bev_embed_ms_flatten = []
        spatial_flatten = []
        for lvl in range(len(bev_embed_ms)):
            B, C, H, W = bev_embed_ms[lvl].shape
            bev_embed_ms_flatten.append(bev_embed_ms[lvl].permute(0, 2, 3, 1).reshape(B, -1, C).permute(1, 0, 2))
            spatial_flatten.append((H, W))
        bev_embed_ms_flatten = torch.cat(bev_embed_ms_flatten, dim=0)
        spatial_flatten = torch.as_tensor(spatial_flatten, dtype=torch.long, device=query_feat_map.device)
        level_start_index = torch.cat((spatial_flatten.new_zeros((1,)), spatial_flatten.prod(1).cumsum(0)[:-1]))
        
        # inter_states, inter_references = self.pts_bbox_head.transformer.decoder(
        #     query=query_feat,
        #     key=None,
        #     value=bev_embed_ms_flatten,
        #     query_pos=query_pos,
        #     reference_points=reference_points,
        #     reg_branches=reg_branches_map,
        #     cls_branches=cls_branches_map,
        #     spatial_shapes=spatial_flatten,
        #     level_start_index=level_start_index,
        #     mlvl_feats=None,
        #     feat_flatten=None,
        #     feat_spatial_shapes=None,
        #     feat_level_start_index=None,
        #     **kwargs,
        # )

        output_map = query_feat_map 
        intermediate_map = []
        intermediate_reference_points_map = []

        kwargs_map['key'] = None
        kwargs_map['value'] = bev_embed_ms_flatten
        kwargs_map['query_pos'] = query_pos_map
        kwargs_map["cls_branches"] = cls_branches_map
        kwargs_map["spatial_shapes"] = spatial_flatten
        kwargs_map["level_start_index"] = level_start_index
        kwargs_map["mlvl_feats"] = None
        kwargs_map["feat_flatten"] = None
        kwargs_map["feat_spatial_shapes"] = None
        kwargs_map["feat_level_start_index"] = None

        for lid, (layer_obj, layer_map) in enumerate(zip(self.obj_pts_bbox_head.transformer.decoder.layers, self.pts_bbox_head.transformer.decoder.layers)):
            
            reference_points_input_obj = reference_points_obj[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            output_obj = layer_obj(
                output_obj,
                reference_points=reference_points_input_obj,
                key_padding_mask=None,
                **kwargs_obj)
            

            reference_points_input_map = reference_points_map[..., :2].unsqueeze(2)  # BS NUM_QUERY NUM_LEVEL 2
            output_map = layer_map(output_map, reference_points=reference_points_input_map, key_padding_mask=None, **kwargs_map)


            residual = output_map 

            output_map = self.cross_attns_map_to_obj[lid](output_map, output_obj, output_obj, need_weights=False)[0]
            for i, l in enumerate(self.ffns_map_to_obj[lid]): 
                output_map = l(output_map)
            output_map = attention_factor*self.norms_map_to_obj[lid](output_map) + residual_factor*residual


            # residual = output_obj 

            # output_obj = self.cross_attns_obj_to_map[lid](output_obj, output_map, output_map, need_weights=False)[0]
            # for i, l in enumerate(self.ffns_obj_to_map[lid]): 
            #     output_obj = l(output_obj)
            # output_obj = attention_factor*self.norms_obj_to_map[lid](output_obj) + residual_factor*residual


            output_obj = output_obj.permute(1, 0, 2)
            output_map = output_map.permute(1, 0, 2)

            if reg_branches_map is not None:
                tmp = reg_branches_map[lid](output_map)

                if output_map.shape[1] < 500:
                    bs, num_query, num_pts, code_size = reference_points_map.shape
                    tmp = tmp.view(bs, num_query, num_pts, code_size)

                new_reference_points = tmp + inverse_sigmoid(reference_points_map)
                new_reference_points = new_reference_points.sigmoid()
                reference_points_map = new_reference_points.detach()

            output_map = output_map.permute(1, 0, 2)
            if self.pts_bbox_head.transformer.decoder.return_intermediate:
                intermediate_map.append(output_map)
                intermediate_reference_points_map.append(reference_points_map)


            if reg_branches_obj is not None:
                tmp = reg_branches_obj[lid](output_obj)

                assert reference_points_obj.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points_obj)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points_obj[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points_obj[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points_obj = new_reference_points.detach()

            output_obj = output_obj.permute(1, 0, 2)
            if self.obj_pts_bbox_head.transformer.decoder.return_intermediate:
                intermediate_obj.append(output_obj)
                intermediate_reference_points_obj.append(reference_points_obj)   

        #-------------------------------------------------------------------------------------------------------------

        if self.pts_bbox_head.transformer.decoder.return_intermediate:
            output_map = torch.stack(intermediate_map)
            reference_points_map = torch.stack(intermediate_reference_points_map)
        
        outputs_map = (bev_embed_map, mask_pred_list, cls_pred_list, depth, output_map, init_reference_out_map, reference_points_map, mask_dict)

        bev_embed, segm_mask_pred_list, segm_cls_scores_list, depth, hs, init_reference, inter_references, mask_dict = outputs_map
        hs = hs.permute(0, 2, 1, 3)
        dn_pad_size = mask_dict["pad_size"] if mask_dict is not None else 0

        outputs_segm_mask_pred_dn = []
        outputs_segm_cls_scores_dn = []

        outputs_segm_mask_pred_one2one = []
        outputs_segm_cls_scores_one2one = []

        outputs_segm_mask_pred_one2many = []
        outputs_segm_cls_scores_one2many = []

        for lvl in range(len(segm_mask_pred_list)):
            outputs_segm_mask_pred_dn.append(segm_mask_pred_list[lvl][:, :dn_pad_size])
            outputs_segm_cls_scores_dn.append(segm_cls_scores_list[lvl][:, :dn_pad_size])

            outputs_segm_mask_pred_one2one.append(segm_mask_pred_list[lvl][:, dn_pad_size : dn_pad_size + self.pts_bbox_head.num_vec_one2one])
            outputs_segm_cls_scores_one2one.append(segm_cls_scores_list[lvl][:, dn_pad_size : dn_pad_size + self.pts_bbox_head.num_vec_one2one])

            outputs_segm_mask_pred_one2many.append(segm_mask_pred_list[lvl][:, dn_pad_size + self.pts_bbox_head.num_vec_one2one :])
            outputs_segm_cls_scores_one2many.append(segm_cls_scores_list[lvl][:, dn_pad_size + self.pts_bbox_head.num_vec_one2one :])

        outputs_segm_mask_pred_dn = torch.stack(outputs_segm_mask_pred_dn)
        outputs_segm_cls_scores_dn = torch.stack(outputs_segm_cls_scores_dn)
        outputs_segm_mask_pred_one2one = torch.stack(outputs_segm_mask_pred_one2one)
        outputs_segm_cls_scores_one2one = torch.stack(outputs_segm_cls_scores_one2one)
        outputs_segm_mask_pred_one2many = torch.stack(outputs_segm_mask_pred_one2many)
        outputs_segm_cls_scores_one2many = torch.stack(outputs_segm_cls_scores_one2many)

        outputs_classes_dn = []
        outputs_coords_dn = []
        outputs_pts_coords_dn = []

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_pts_coords_one2one = []

        outputs_classes_one2many = []
        outputs_coords_one2many = []
        outputs_pts_coords_one2many = []

        num_vec = hs.shape[2] // self.pts_bbox_head.num_pts_per_vec
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # import pdb;pdb.set_trace()
                reference = init_reference[..., 0:2] if not self.pts_bbox_head.z_cfg["gt_z_flag"] else init_reference[..., 0:3]
            else:
                reference = inter_references[lvl - 1][..., 0:2] if not self.pts_bbox_head.z_cfg["gt_z_flag"] else inter_references[lvl - 1][..., 0:3]
            reference = inverse_sigmoid(reference)
            outputs_class = self.pts_bbox_head.cls_branches[lvl](hs[lvl].view(bs, num_vec, self.pts_bbox_head.num_pts_per_vec, -1).mean(2))
            tmp = self.pts_bbox_head.reg_branches[lvl](hs[lvl])
            tmp = tmp[..., 0:2] if not self.pts_bbox_head.z_cfg["gt_z_flag"] else tmp[..., 0:3]
            tmp += reference

            tmp = tmp.sigmoid()  # cx,cy,w,h

            outputs_coord, outputs_pts_coord = self.pts_bbox_head.transform_box(tmp, num_vec=num_vec)

            outputs_classes_one2one.append(outputs_class[:, dn_pad_size : dn_pad_size + self.pts_bbox_head.num_vec_one2one])
            outputs_coords_one2one.append(outputs_coord[:, dn_pad_size : dn_pad_size + self.pts_bbox_head.num_vec_one2one])
            outputs_pts_coords_one2one.append(outputs_pts_coord[:, dn_pad_size : dn_pad_size + self.pts_bbox_head.num_vec_one2one])

            outputs_classes_dn.append(outputs_class[:, :dn_pad_size])
            outputs_coords_dn.append(outputs_coord[:, :dn_pad_size])
            outputs_pts_coords_dn.append(outputs_pts_coord[:, :dn_pad_size])

            outputs_classes_one2many.append(outputs_class[:, dn_pad_size + self.pts_bbox_head.num_vec_one2one :])
            outputs_coords_one2many.append(outputs_coord[:, dn_pad_size + self.pts_bbox_head.num_vec_one2one :])
            outputs_pts_coords_one2many.append(outputs_pts_coord[:, dn_pad_size + self.pts_bbox_head.num_vec_one2one :])

        outputs_classes_dn = torch.stack(outputs_classes_dn)
        outputs_coords_dn = torch.stack(outputs_coords_dn)
        outputs_pts_coords_dn = torch.stack(outputs_pts_coords_dn)

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_pts_coords_one2one = torch.stack(outputs_pts_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        outputs_pts_coords_one2many = torch.stack(outputs_pts_coords_one2many)

        if self.pts_bbox_head.dn_enabled and self.pts_bbox_head.training:
            mask_dict["output_known_lbs_bboxes"] = (
                outputs_segm_cls_scores_dn,
                outputs_segm_mask_pred_dn,
                outputs_classes_dn,
                outputs_coords_dn,
                outputs_pts_coords_dn,
            )

        outputs_seg = None
        outputs_pv_seg = None
        if self.pts_bbox_head.aux_seg["use_aux_seg"]:
            seg_bev_embed = bev_embed.permute(1, 0, 2).reshape(bs, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, -1).permute(0, 3, 1, 2).contiguous()
            if self.pts_bbox_head.aux_seg["bev_seg"]:
                outputs_seg = self.pts_bbox_head.seg_head(seg_bev_embed)
            bs, num_cam, embed_dims, feat_h, feat_w = pts_feats[-1].shape
            if self.pts_bbox_head.aux_seg["pv_seg"]:
                outputs_pv_seg = self.pts_bbox_head.pv_seg_head(pts_feats[-1].flatten(0, 1))
                outputs_pv_seg = outputs_pv_seg.view(bs, num_cam, -1, feat_h, feat_w)

        if self.pts_bbox_head.tr_cls_join and self.pts_bbox_head.training:
            outputs_classes_one2one = outputs_classes_one2one + outputs_segm_cls_scores_one2one[-1:][..., :3].contiguous()

        if self.pts_bbox_head.cls_join and not self.pts_bbox_head.training:
            outputs_classes_one2one = outputs_classes_one2one + outputs_segm_cls_scores_one2one[-1:][..., :3].contiguous()
        outs_map = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes_one2one,
            "all_bbox_preds": outputs_coords_one2one,
            "all_pts_preds": outputs_pts_coords_one2one,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
            "depth": depth,
            "seg": outputs_seg,
            "segm_mask_preds": outputs_segm_mask_pred_one2one,
            "segm_cls_scores": outputs_segm_cls_scores_one2one,
            "pv_seg": outputs_pv_seg,
            "one2many_outs": dict(
                all_cls_scores=outputs_classes_one2many,
                all_bbox_preds=outputs_coords_one2many,
                all_pts_preds=outputs_pts_coords_one2many,
                enc_bbox_preds=None,
                enc_pts_preds=None,
                segm_mask_preds=outputs_segm_mask_pred_one2many,
                segm_cls_scores=outputs_segm_cls_scores_one2many,
                seg=None,
                pv_seg=None,
            ),
            "dn_mask_dict": mask_dict,
        }

        #-------------------------------------------------------------------

        if self.obj_pts_bbox_head.transformer.decoder.return_intermediate:
            inter_states_obj = torch.stack(intermediate_obj)
            inter_references_out_obj = torch.stack(intermediate_reference_points_obj)

        outputs_obj = (bev_embed_obj, inter_states_obj, init_reference_out_obj, inter_references_out_obj)


        bev_embed, hs, init_reference, inter_references = outputs_obj
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.obj_pts_bbox_head.cls_branches[lvl](hs[lvl])
            tmp = self.obj_pts_bbox_head.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.obj_pts_bbox_head.pc_range[3] -
                             self.obj_pts_bbox_head.pc_range[0]) + self.obj_pts_bbox_head.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.obj_pts_bbox_head.pc_range[4] -
                             self.obj_pts_bbox_head.pc_range[1]) + self.obj_pts_bbox_head.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.obj_pts_bbox_head.pc_range[5] -
                             self.obj_pts_bbox_head.pc_range[2]) + self.obj_pts_bbox_head.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs_obj = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }




        losses = dict()
        
        #Mask2Map head losses
        depth = outs_map.pop("depth")
        # calculate depth loss
        if gt_depth is not None:
            loss_depth = self.pts_bbox_head.transformer.encoder.get_depth_loss(gt_depth, depth)
            if digit_version(TORCH_VERSION) >= digit_version("1.8"):
                loss_depth = torch.nan_to_num(loss_depth)
            losses.update(loss_depth=loss_depth)

        gt_vecs_list = copy.deepcopy(gt_bboxes_3d)
        device = gt_labels_3d[0].device

        #labels for all map instances
        gt_labels_list = [gt_bboxes.gt_labels.to(device) for gt_bboxes in gt_vecs_list]                                     
        #2D bboxes for all map instances that cover all points for that instance
        gt_bboxes_list = [gt_bboxes.bbox_condi.to(device) for gt_bboxes in gt_vecs_list]
        #ordered point sets for all map instances
        gt_pts_list = [gt_bboxes.fixed_num_sampled_points_condi.to(device) for gt_bboxes in gt_vecs_list]
        #probbly the noised map instances for the denoising loss 
        gt_shifts_pts_list = [gt_bboxes.shift_fixed_num_sampled_points_v2_condi.to(device) for gt_bboxes in gt_vecs_list]

        gt_segms_list = [gt_bboxes.instance_segments_condi.to(device) for gt_bboxes in gt_vecs_list]
        loss_inputs = [gt_bboxes_list, gt_labels_list, gt_pts_list, gt_seg_mask, gt_pv_seg_mask, gt_shifts_pts_list, gt_segms_list, outs_map]

        losses_pts = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        losses.update(losses_pts)


        #BEVFormer head losses
        if "seg_preds" not in outs_obj.keys():
            obj_gt_labels_3d = [torch.tensor(label, device='cuda') for label in obj_gt_labels_3d]
            det_loss_inputs = [obj_gt_bboxes_3d, obj_gt_labels_3d, outs_obj]
            det_losses = self.obj_pts_bbox_head.loss(*det_loss_inputs, img_metas=img_metas)
            for k,v in det_losses.items(): 
                losses[f'obj_{k}'] = 3.*v #scale the det losses to match the magnitude og the Mask2Map losses 
        else: 
            raise(Exception("The BEVFormer head should only perform OD"))
        
        self.num_iterations_done += 1
        if self.num_iterations_done%50==0: 
            print(f'num iterations done: {self.num_iterations_done}')
            print(f'attention factor: {attention_factor}')
            print(f'residual factor: {residual_factor}')

        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated."""
        self.eval()

        with torch.no_grad():
            prev_bev = {
                'obj': None,
                'map': None
            }
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                #prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True)

                # #Get the encoder features from both heads: 
                #BEVFormer head
                obj_encoder_feats = self.obj_pts_bbox_head(img_feats, img_metas, prev_bev['obj'], only_bev=True)

                #Mask2Map head
                map_encoder_feats = self.pts_bbox_head(img_feats, None, img_metas, prev_bev['map'], only_bev=True)

                prev_bev = {
                    'obj': obj_encoder_feats,
                    'map': map_encoder_feats
                }

            self.train()
            return prev_bev

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    #@auto_fp16(apply_to=("points"), out_fp32=True)
    def extract_lidar_feat(self, points):
        feats, coords, sizes = self.voxelize(points)
        # voxel_features = self.lidar_modal_extractor["voxel_encoder"](feats, sizes, coords)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)

        return lidar_feat

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=("img", "points", "prev_bev"))
    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
        gt_depth=None,
        gt_seg_mask=None,
        gt_pv_seg_mask=None,
        obj_gt_bboxes_3d=None,
        obj_gt_labels_3d=None, 
        obj_gt_names=None
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        lidar_feat = None
        if self.modality == "fusion":
            lidar_feat = self.extract_lidar_feat(points)

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None
        img_metas = [each[len_queue - 1] for each in img_metas]

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats,
            lidar_feat,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            prev_bev,
            gt_depth,
            gt_seg_mask,
            gt_pv_seg_mask,
            obj_gt_bboxes_3d,
            obj_gt_labels_3d, 
            obj_gt_names
        )

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(img_metas[0], img[0], points[0], prev_bev=self.prev_frame_info["prev_bev"], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev
        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to("cpu"),
        )

        if attrs is not None:
            result_dict["attrs_3d"] = attrs.cpu()

        return result_dict

    def pred2result_dist(self, bboxes, scores, labels, pts, pts_cls, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to("cpu"),
            pts_cls_3d=pts_cls.to("cpu"),
        )

        if attrs is not None:
            result_dict["attrs_3d"] = attrs.cpu()

        return result_dict

    def simple_test_pts(self, pts_feats, lidar_feat, img_metas, prev_bev=None, rescale=False):
        
        #determine the factors to use for the inter-task attention outputs and the residual connections aruond the attention blocks 
        runner = get_runner()
        current_epoch = runner.epoch  #int(self.num_iterations_done/7033) #only correct for BS of 4 
        current_epoch = min(current_epoch, 3) 
        attention_factor = (current_epoch/11)*0.5
        residual_factor = 1-attention_factor


        """Test function"""
        #Get the encoder features from both heads: 
         #BEVFormer head
        obj_encoder_feats = self.obj_pts_bbox_head(pts_feats, img_metas, prev_bev['pts_bbox_obj'] if prev_bev is not None else None, only_bev=True)
        residual_obj = obj_encoder_feats

        #Mask2Map head
        map_encoder_feats = self.pts_bbox_head(pts_feats, lidar_feat, img_metas, prev_bev['pts_bbox'] if prev_bev is not None else None, only_bev=True)
        residual_map = map_encoder_feats['bev']

        #ITSA between the two tasks' encoder features
        #TODO: might work better with two separate def attn blocks, one for each task, also gradient scaling might be necessary here 
        bs = obj_encoder_feats.shape[0]
        reference_points = self.reference_points
        for i in range(bs-1): 
            reference_points = torch.cat((reference_points, self.reference_points), dim=0)
        refined_feats = self.deformable_latent_fusion(query = torch.cat((obj_encoder_feats, map_encoder_feats['bev']), dim=1), 
                                                      input_flatten = torch.cat((obj_encoder_feats, map_encoder_feats['bev']), dim=1), 
                                                      input_spatial_shapes = torch.tensor([[self.obj_pts_bbox_head.bev_h, self.obj_pts_bbox_head.bev_w], [200, 100]]).cuda(),
                                                      input_level_start_index = torch.tensor([0,self.obj_pts_bbox_head.bev_h*self.obj_pts_bbox_head.bev_w]).cuda(),
                                                      reference_points = reference_points)

        obj_encoder_feats = residual_obj
        map_encoder_feats['bev'] = attention_factor*refined_feats[:, self.obj_pts_bbox_head.bev_h*self.obj_pts_bbox_head.bev_w:, ...] + residual_factor*residual_map

        if isinstance(self.pts_bbox_head, Mask2MapHead_1Phase):
            #Get the decdoere outputs given the refined encoder features for both heads:
            #BEVFormer head
            outs_obj = self.obj_pts_bbox_head(pts_feats, img_metas, prev_bev, bev_embed=obj_encoder_feats)
            #Mask2Map head
            outs_map = self.pts_bbox_head(pts_feats, lidar_feat, img_metas, prev_bev, encoder_feats=map_encoder_feats)
        else: #phase 2
            #Get the decoder outputs given the refined encoder features for both heads:
            #BEVFormer head
            #-------------------------------------------------------------------------------------------------
            #outs_obj = self.obj_pts_bbox_head(pts_feats, img_metas, prev_bev, bev_embed=obj_encoder_feats)

            bs, num_cam, _, _, _ = pts_feats[0].shape
            dtype = pts_feats[0].dtype
            object_query_embeds = self.obj_pts_bbox_head.query_embedding.weight.to(dtype)
            bev_queries = self.obj_pts_bbox_head.bev_embedding.weight.to(dtype)

            bev_mask = torch.zeros((bs, self.obj_pts_bbox_head.bev_h, self.obj_pts_bbox_head.bev_w),
                                device=bev_queries.device).to(dtype)
            bev_pos = self.obj_pts_bbox_head.positional_encoding(bev_mask).to(dtype)

            # outputs = self.obj_pts_bbox_head.transformer(
            #     pts_feats,
            #     bev_queries,
            #     object_query_embeds,
            #     self.obj_pts_bbox_head.bev_h,
            #     self.obj_pts_bbox_head.bev_w,
            #     grid_length=(self.obj_pts_bbox_head.real_h / self.obj_pts_bbox_head.bev_h,
            #                     self.obj_pts_bbox_head.real_w / self.obj_pts_bbox_head.bev_w),
            #     bev_pos=bev_pos,
            #     reg_branches=self.obj_pts_bbox_head.reg_branches if self.obj_pts_bbox_head.with_box_refine else None,  # noqa:E501
            #     cls_branches=self.obj_pts_bbox_head.cls_branches if self.obj_pts_bbox_head.as_two_stage else None,
            #     img_metas=img_metas,
            #     prev_bev=prev_bev,
            #     bev_embed=obj_encoder_feats
            # )

            bs = pts_feats[0].size(0)
            query_pos_obj, query_obj = torch.split(
                object_query_embeds, self.obj_pts_bbox_head.transformer.embed_dims, dim=1)
            query_pos_obj = query_pos_obj.unsqueeze(0).expand(bs, -1, -1)
            query_obj = query_obj.unsqueeze(0).expand(bs, -1, -1)
            reference_points_obj = self.obj_pts_bbox_head.transformer.reference_points(query_pos_obj)
            reference_points_obj = reference_points_obj.sigmoid()
            init_reference_out_obj = reference_points_obj

            query_obj = query_obj.permute(1, 0, 2)
            query_pos_obj = query_pos_obj.permute(1, 0, 2)
            bev_embed_obj = obj_encoder_feats.permute(1, 0, 2)

            reg_branches_obj=self.obj_pts_bbox_head.reg_branches if self.obj_pts_bbox_head.with_box_refine else None  # noqa:E501
            cls_branches_obj=self.obj_pts_bbox_head.cls_branches if self.obj_pts_bbox_head.as_two_stage else None

            # inter_states, inter_references = self.obj_pts_bbox_head.transformer.decoder(
            #     query=query,
            #     key=None,
            #     value=bev_embed,
            #     query_pos=query_pos,
            #     reference_points=reference_points,
            #     reg_branches=reg_branches,
            #     cls_branches=cls_branches,
            #     spatial_shapes=torch.tensor([[self.obj_pts_bbox_head.bev_h, self.obj_pts_bbox_head.bev_w]], device=query.device),
            #     level_start_index=torch.tensor([0], device=query.device),
            #     img_metas=img_metas)

            output_obj = query_obj
            intermediate_obj = []
            intermediate_reference_points_obj = []
            kwargs_obj = dict(key=None, 
                        value=bev_embed_obj,
                        query_pos=query_pos_obj, 
                        cls_branches=cls_branches_obj, 
                        spatial_shapes=torch.tensor([[self.obj_pts_bbox_head.bev_h, self.obj_pts_bbox_head.bev_w]], device=query_obj.device),
                        level_start_index=torch.tensor([0], device=query_obj.device),
                        img_metas=img_metas
                        )
            # for lid, layer in enumerate(self.obj_pts_bbox_head.transformer.decoder.layers):

            #     reference_points_input_obj = reference_points_obj[..., :2].unsqueeze(
            #         2)  # BS NUM_QUERY NUM_LEVEL 2
            #     output_obj = layer(
            #         output_obj,
            #         reference_points=reference_points_input_obj,
            #         key_padding_mask=None,
            #         **kwargs_obj)
            #     output_obj = output_obj.permute(1, 0, 2)

            #     if reg_branches_obj is not None:
            #         tmp = reg_branches_obj[lid](output_obj)

            #         assert reference_points_obj.shape[-1] == 3

            #         new_reference_points = torch.zeros_like(reference_points_obj)
            #         new_reference_points[..., :2] = tmp[
            #             ..., :2] + inverse_sigmoid(reference_points_obj[..., :2])
            #         new_reference_points[..., 2:3] = tmp[
            #             ..., 4:5] + inverse_sigmoid(reference_points_obj[..., 2:3])

            #         new_reference_points = new_reference_points.sigmoid()

            #         reference_points_obj = new_reference_points.detach()

            #     output_obj = output_obj.permute(1, 0, 2)
            #     if self.obj_pts_bbox_head.transformer.decoder.return_intermediate:
            #         intermediate_obj.append(output_obj)
            #         intermediate_reference_points_obj.append(reference_points_obj)

            #     #TODO: for each refinement layer, the queries in output need to be 
            #     # cross-attended with the map prediction queries of the corresponding refinement layer 

            
            
            #-------------------------------------------------------------------------------------------------

            #Mask2Map head
            #TODO: this is for the pts_bbox_head for phase 2, phase 1 needs to be handled separately 
            # (does phase 1 even compute the MaskGuidedMapDecoder?, otherwise only phase 2 needs to be dealt with)
            #-------------------------------------------------------------------------------------------------
            #outs = self.pts_bbox_head(pts_feats, lidar_feat, img_metas, prev_bev, encoder_feats=map_encoder_feats)

            num_vec = self.pts_bbox_head.num_vec_one2one

            bs, num_cam, _, _, _ = pts_feats[0].shape
            dtype = pts_feats[0].dtype

            pts_query = self.pts_bbox_head.pts_embedding.weight.unsqueeze(0)

            bev_queries = None
            bev_mask = None
            bev_pos = None

            # make attn mask
            """ attention mask to prevent information leakage
            """
            self_attn_mask = torch.zeros([num_vec, num_vec]).bool().to(pts_feats[0].device)
            self_attn_mask[self.pts_bbox_head.num_vec_one2one :, 0 : self.pts_bbox_head.num_vec_one2one] = True
            self_attn_mask[0 : self.pts_bbox_head.num_vec_one2one, self.pts_bbox_head.num_vec_one2one :] = True

            # outputs = self.pts_bbox_head.transformer(
            #     pts_feats,
            #     lidar_feat,
            #     bev_queries,
            #     pts_query,
            #     self.pts_bbox_head.bev_h,
            #     self.pts_bbox_head.bev_w,
            #     grid_length=(self.pts_bbox_head.real_h / self.pts_bbox_head.bev_h, self.pts_bbox_head.real_w / self.pts_bbox_head.bev_w),
            #     bev_pos=bev_pos,
            #     reg_branches=(self.pts_bbox_head.reg_branches if self.pts_bbox_head.with_box_refine else None),  # noqa:E501
            #     cls_branches=self.pts_bbox_head.cls_branches if self.pts_bbox_head.as_two_stage else None,
            #     img_metas=img_metas,
            #     prev_bev=prev_bev,
            #     self_attn_mask=self_attn_mask,
            #     num_vec=num_vec,
            #     num_pts_per_vec=self.pts_bbox_head.num_pts_per_vec,
            #     encoder_feats=map_encoder_feats
            # )

            grid_length=(self.pts_bbox_head.real_h / self.pts_bbox_head.bev_h, self.pts_bbox_head.real_w / self.pts_bbox_head.bev_w)
            kwargs_map = dict(img_metas=img_metas,
                    self_attn_mask=self_attn_mask,
                    num_vec=num_vec, 
                    num_pts_per_vec=self.pts_bbox_head.num_pts_per_vec,
                    encoder_feats=map_encoder_feats)
            (mask_pred_list, cls_pred_list, mask_aware_query_feat, bev_embed_map, bev_embed_ms, depth, dn_information, kwargs_map) = self.pts_bbox_head.transformer.IMPNet(
                pts_feats, lidar_feat, bev_queries, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, grid_length=grid_length, bev_pos=bev_pos, prev_bev=prev_bev, **kwargs_map
            )

            # MMPNet
            mask_dict, known_masks, known_bid, map_known_indice = dn_information

            mask_aware_query_feat, mask_aware_query_pos = self.pts_bbox_head.transformer.PositionalQueryGenerator(
                mask_aware_query_feat,
                pts_query,
                mask_pred_list[-1],
                known_masks=known_masks,
                known_bid=known_bid,
                map_known_indice=map_known_indice,
            )

            query_map, key_map, value_map, mask_aware_query, mask_aware_query_norm = self.pts_bbox_head.transformer.GeometricFeatureExtractor(
                mask_pred_list,
                mask_aware_query_feat,
                mask_aware_query_pos,
                bev_embed_map,
            )

            reg_branches_map=(self.pts_bbox_head.reg_branches if self.pts_bbox_head.with_box_refine else None)  # noqa:E501
            cls_branches_map=self.pts_bbox_head.cls_branches if self.pts_bbox_head.as_two_stage else None
            
            # inter_states, inter_references_out, init_reference_out = self.pts_bbox_head.transformer.MaskGuidedMapDecoder(
            #     query,
            #     key,
            #     value,
            #     bev_embed_ms,
            #     mask_aware_query,
            #     mask_aware_query_norm,
            #     reg_branches_map,
            #     cls_branches_map,
            #     **kwargs,
            # )

            bs = mask_aware_query.shape[0]
            q = query_map.shape[1]//bs
            attn_output, _ = self.pts_bbox_head.transformer.m_attn(query_map, key_map, value_map)
            attn_output = attn_output.permute(1, 0, 2)
            attn_output = attn_output.reshape(bs, q, -1, self.pts_bbox_head.transformer.embed_dims * 2)
            attn_output = attn_output.reshape(bs, -1, self.pts_bbox_head.transformer.embed_dims * 2)

            gate = torch.sigmoid(self.pts_bbox_head.transformer.lin_ih(attn_output) + self.pts_bbox_head.transformer.lin_hh(mask_aware_query_norm))
            query_map = mask_aware_query + self.pts_bbox_head.transformer.out_proj(attn_output + gate * (self.pts_bbox_head.transformer.lin_self(mask_aware_query_norm) - attn_output))
            query_map = query_map + self.pts_bbox_head.transformer.mlp2(self.pts_bbox_head.transformer.norm2(query_map))

            query_pos_map, query_feat_map = torch.split(query_map, self.pts_bbox_head.transformer.embed_dims, dim=-1)

            reference_points_map = self.pts_bbox_head.transformer.reference_points(query_pos_map)
            reference_points_map = reference_points_map.sigmoid()
            init_reference_out_map = reference_points_map

            query_feat_map = query_feat_map.permute(1, 0, 2) 
            query_pos_map = query_pos_map.permute(1, 0, 2)
            bev_embed_ms_flatten = []
            spatial_flatten = []
            for lvl in range(len(bev_embed_ms)):
                B, C, H, W = bev_embed_ms[lvl].shape
                bev_embed_ms_flatten.append(bev_embed_ms[lvl].permute(0, 2, 3, 1).reshape(B, -1, C).permute(1, 0, 2))
                spatial_flatten.append((H, W))
            bev_embed_ms_flatten = torch.cat(bev_embed_ms_flatten, dim=0)
            spatial_flatten = torch.as_tensor(spatial_flatten, dtype=torch.long, device=query_feat_map.device)
            level_start_index = torch.cat((spatial_flatten.new_zeros((1,)), spatial_flatten.prod(1).cumsum(0)[:-1]))
            
            # inter_states, inter_references = self.pts_bbox_head.transformer.decoder(
            #     query=query_feat,
            #     key=None,
            #     value=bev_embed_ms_flatten,
            #     query_pos=query_pos,
            #     reference_points=reference_points,
            #     reg_branches=reg_branches_map,
            #     cls_branches=cls_branches_map,
            #     spatial_shapes=spatial_flatten,
            #     level_start_index=level_start_index,
            #     mlvl_feats=None,
            #     feat_flatten=None,
            #     feat_spatial_shapes=None,
            #     feat_level_start_index=None,
            #     **kwargs,
            # )

            output_map = query_feat_map 
            intermediate_map = []
            intermediate_reference_points_map = []

            kwargs_map['key'] = None
            kwargs_map['value'] = bev_embed_ms_flatten
            kwargs_map['query_pos'] = query_pos_map
            kwargs_map["cls_branches"] = cls_branches_map
            kwargs_map["spatial_shapes"] = spatial_flatten
            kwargs_map["level_start_index"] = level_start_index
            kwargs_map["mlvl_feats"] = None
            kwargs_map["feat_flatten"] = None
            kwargs_map["feat_spatial_shapes"] = None
            kwargs_map["feat_level_start_index"] = None

            for lid, (layer_obj, layer_map) in enumerate(zip(self.obj_pts_bbox_head.transformer.decoder.layers, self.pts_bbox_head.transformer.decoder.layers)):
                
                reference_points_input_obj = reference_points_obj[..., :2].unsqueeze(
                    2)  # BS NUM_QUERY NUM_LEVEL 2
                output_obj = layer_obj(
                    output_obj,
                    reference_points=reference_points_input_obj,
                    key_padding_mask=None,
                    **kwargs_obj)
                

                reference_points_input_map = reference_points_map[..., :2].unsqueeze(2)  # BS NUM_QUERY NUM_LEVEL 2
                output_map = layer_map(output_map, reference_points=reference_points_input_map, key_padding_mask=None, **kwargs_map)


                residual = output_map 

                output_map = self.cross_attns_map_to_obj[lid](output_map, output_obj, output_obj, need_weights=False)[0]
                for i, l in enumerate(self.ffns_map_to_obj[lid]): 
                    output_map = l(output_map) 
                output_map = attention_factor*self.norms_map_to_obj[lid](output_map) + residual_factor*residual

                
                # residual = output_obj 

                # output_obj = self.cross_attns_obj_to_map[lid](output_obj, output_map, output_map, need_weights=False)[0]
                # for i, l in enumerate(self.ffns_obj_to_map[lid]): 
                #     output_obj = l(output_obj)
                # output_obj = attention_factor*self.norms_obj_to_map[lid](output_obj) + residual_factor*residual


                output_obj = output_obj.permute(1, 0, 2)
                output_map = output_map.permute(1, 0, 2)

                if reg_branches_map is not None:
                    tmp = reg_branches_map[lid](output_map)

                    if output_map.shape[1] < 500:
                        bs, num_query, num_pts, code_size = reference_points_map.shape
                        tmp = tmp.view(bs, num_query, num_pts, code_size)

                    new_reference_points = tmp + inverse_sigmoid(reference_points_map)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points_map = new_reference_points.detach()

                output_map = output_map.permute(1, 0, 2)
                if self.pts_bbox_head.transformer.decoder.return_intermediate:
                    intermediate_map.append(output_map)
                    intermediate_reference_points_map.append(reference_points_map)


                if reg_branches_obj is not None:
                    tmp = reg_branches_obj[lid](output_obj)

                    assert reference_points_obj.shape[-1] == 3

                    new_reference_points = torch.zeros_like(reference_points_obj)
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points_obj[..., :2])
                    new_reference_points[..., 2:3] = tmp[
                        ..., 4:5] + inverse_sigmoid(reference_points_obj[..., 2:3])

                    new_reference_points = new_reference_points.sigmoid()

                    reference_points_obj = new_reference_points.detach()

                output_obj = output_obj.permute(1, 0, 2)
                if self.obj_pts_bbox_head.transformer.decoder.return_intermediate:
                    intermediate_obj.append(output_obj)
                    intermediate_reference_points_obj.append(reference_points_obj)   

            #-------------------------------------------------------------------------------------------------------------

            if self.pts_bbox_head.transformer.decoder.return_intermediate:
                output_map = torch.stack(intermediate_map)
                reference_points_map = torch.stack(intermediate_reference_points_map)
            
            outputs_map = (bev_embed_map, mask_pred_list, cls_pred_list, depth, output_map, init_reference_out_map, reference_points_map, mask_dict)

            bev_embed, segm_mask_pred_list, segm_cls_scores_list, depth, hs, init_reference, inter_references, mask_dict = outputs_map
            hs = hs.permute(0, 2, 1, 3)
            dn_pad_size = mask_dict["pad_size"] if mask_dict is not None else 0

            outputs_segm_mask_pred_dn = []
            outputs_segm_cls_scores_dn = []

            outputs_segm_mask_pred_one2one = []
            outputs_segm_cls_scores_one2one = []

            outputs_segm_mask_pred_one2many = []
            outputs_segm_cls_scores_one2many = []

            for lvl in range(len(segm_mask_pred_list)):
                outputs_segm_mask_pred_dn.append(segm_mask_pred_list[lvl][:, :dn_pad_size])
                outputs_segm_cls_scores_dn.append(segm_cls_scores_list[lvl][:, :dn_pad_size])

                outputs_segm_mask_pred_one2one.append(segm_mask_pred_list[lvl][:, dn_pad_size : dn_pad_size + self.pts_bbox_head.num_vec_one2one])
                outputs_segm_cls_scores_one2one.append(segm_cls_scores_list[lvl][:, dn_pad_size : dn_pad_size + self.pts_bbox_head.num_vec_one2one])

                outputs_segm_mask_pred_one2many.append(segm_mask_pred_list[lvl][:, dn_pad_size + self.pts_bbox_head.num_vec_one2one :])
                outputs_segm_cls_scores_one2many.append(segm_cls_scores_list[lvl][:, dn_pad_size + self.pts_bbox_head.num_vec_one2one :])

            outputs_segm_mask_pred_dn = torch.stack(outputs_segm_mask_pred_dn)
            outputs_segm_cls_scores_dn = torch.stack(outputs_segm_cls_scores_dn)
            outputs_segm_mask_pred_one2one = torch.stack(outputs_segm_mask_pred_one2one)
            outputs_segm_cls_scores_one2one = torch.stack(outputs_segm_cls_scores_one2one)
            outputs_segm_mask_pred_one2many = torch.stack(outputs_segm_mask_pred_one2many)
            outputs_segm_cls_scores_one2many = torch.stack(outputs_segm_cls_scores_one2many)

            outputs_classes_dn = []
            outputs_coords_dn = []
            outputs_pts_coords_dn = []

            outputs_classes_one2one = []
            outputs_coords_one2one = []
            outputs_pts_coords_one2one = []

            outputs_classes_one2many = []
            outputs_coords_one2many = []
            outputs_pts_coords_one2many = []

            num_vec = hs.shape[2] // self.pts_bbox_head.num_pts_per_vec
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    # import pdb;pdb.set_trace()
                    reference = init_reference[..., 0:2] if not self.pts_bbox_head.z_cfg["gt_z_flag"] else init_reference[..., 0:3]
                else:
                    reference = inter_references[lvl - 1][..., 0:2] if not self.pts_bbox_head.z_cfg["gt_z_flag"] else inter_references[lvl - 1][..., 0:3]
                reference = inverse_sigmoid(reference)
                outputs_class = self.pts_bbox_head.cls_branches[lvl](hs[lvl].view(bs, num_vec, self.pts_bbox_head.num_pts_per_vec, -1).mean(2))
                tmp = self.pts_bbox_head.reg_branches[lvl](hs[lvl])
                tmp = tmp[..., 0:2] if not self.pts_bbox_head.z_cfg["gt_z_flag"] else tmp[..., 0:3]
                tmp += reference

                tmp = tmp.sigmoid()  # cx,cy,w,h

                outputs_coord, outputs_pts_coord = self.pts_bbox_head.transform_box(tmp, num_vec=num_vec)

                outputs_classes_one2one.append(outputs_class[:, dn_pad_size : dn_pad_size + self.pts_bbox_head.num_vec_one2one])
                outputs_coords_one2one.append(outputs_coord[:, dn_pad_size : dn_pad_size + self.pts_bbox_head.num_vec_one2one])
                outputs_pts_coords_one2one.append(outputs_pts_coord[:, dn_pad_size : dn_pad_size + self.pts_bbox_head.num_vec_one2one])

                outputs_classes_dn.append(outputs_class[:, :dn_pad_size])
                outputs_coords_dn.append(outputs_coord[:, :dn_pad_size])
                outputs_pts_coords_dn.append(outputs_pts_coord[:, :dn_pad_size])

                outputs_classes_one2many.append(outputs_class[:, dn_pad_size + self.pts_bbox_head.num_vec_one2one :])
                outputs_coords_one2many.append(outputs_coord[:, dn_pad_size + self.pts_bbox_head.num_vec_one2one :])
                outputs_pts_coords_one2many.append(outputs_pts_coord[:, dn_pad_size + self.pts_bbox_head.num_vec_one2one :])

            outputs_classes_dn = torch.stack(outputs_classes_dn)
            outputs_coords_dn = torch.stack(outputs_coords_dn)
            outputs_pts_coords_dn = torch.stack(outputs_pts_coords_dn)

            outputs_classes_one2one = torch.stack(outputs_classes_one2one)
            outputs_coords_one2one = torch.stack(outputs_coords_one2one)
            outputs_pts_coords_one2one = torch.stack(outputs_pts_coords_one2one)

            outputs_classes_one2many = torch.stack(outputs_classes_one2many)
            outputs_coords_one2many = torch.stack(outputs_coords_one2many)
            outputs_pts_coords_one2many = torch.stack(outputs_pts_coords_one2many)

            if self.pts_bbox_head.dn_enabled and self.pts_bbox_head.training:
                mask_dict["output_known_lbs_bboxes"] = (
                    outputs_segm_cls_scores_dn,
                    outputs_segm_mask_pred_dn,
                    outputs_classes_dn,
                    outputs_coords_dn,
                    outputs_pts_coords_dn,
                )

            outputs_seg = None
            outputs_pv_seg = None
            if self.pts_bbox_head.aux_seg["use_aux_seg"]:
                seg_bev_embed = bev_embed.permute(1, 0, 2).reshape(bs, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, -1).permute(0, 3, 1, 2).contiguous()
                if self.pts_bbox_head.aux_seg["bev_seg"]:
                    outputs_seg = self.pts_bbox_head.seg_head(seg_bev_embed)
                bs, num_cam, embed_dims, feat_h, feat_w = pts_feats[-1].shape
                if self.pts_bbox_head.aux_seg["pv_seg"]:
                    outputs_pv_seg = self.pts_bbox_head.pv_seg_head(pts_feats[-1].flatten(0, 1))
                    outputs_pv_seg = outputs_pv_seg.view(bs, num_cam, -1, feat_h, feat_w)

            if self.pts_bbox_head.tr_cls_join and self.pts_bbox_head.training:
                outputs_classes_one2one = outputs_classes_one2one + outputs_segm_cls_scores_one2one[-1:][..., :3].contiguous()

            if self.pts_bbox_head.cls_join and not self.pts_bbox_head.training:
                outputs_classes_one2one = outputs_classes_one2one + outputs_segm_cls_scores_one2one[-1:][..., :3].contiguous()
            outs_map = {
                "bev_embed": bev_embed,
                "all_cls_scores": outputs_classes_one2one,
                "all_bbox_preds": outputs_coords_one2one,
                "all_pts_preds": outputs_pts_coords_one2one,
                "enc_bbox_preds": None,
                "enc_pts_preds": None,
                "depth": depth,
                "seg": outputs_seg,
                "segm_mask_preds": outputs_segm_mask_pred_one2one,
                "segm_cls_scores": outputs_segm_cls_scores_one2one,
                "pv_seg": outputs_pv_seg,
                "one2many_outs": dict(
                    all_cls_scores=outputs_classes_one2many,
                    all_bbox_preds=outputs_coords_one2many,
                    all_pts_preds=outputs_pts_coords_one2many,
                    enc_bbox_preds=None,
                    enc_pts_preds=None,
                    segm_mask_preds=outputs_segm_mask_pred_one2many,
                    segm_cls_scores=outputs_segm_cls_scores_one2many,
                    seg=None,
                    pv_seg=None,
                ),
                "dn_mask_dict": mask_dict,
            }

            #-------------------------------------------------------------------

            if self.obj_pts_bbox_head.transformer.decoder.return_intermediate:
                inter_states_obj = torch.stack(intermediate_obj)
                inter_references_out_obj = torch.stack(intermediate_reference_points_obj)

            outputs_obj = (bev_embed_obj, inter_states_obj, init_reference_out_obj, inter_references_out_obj)


            bev_embed, hs, init_reference, inter_references = outputs_obj
            hs = hs.permute(0, 2, 1, 3)
            outputs_classes = []
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                outputs_class = self.obj_pts_bbox_head.cls_branches[lvl](hs[lvl])
                tmp = self.obj_pts_bbox_head.reg_branches[lvl](hs[lvl])

                # TODO: check the shape of reference
                assert reference.shape[-1] == 3
                tmp[..., 0:2] += reference[..., 0:2]
                tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
                tmp[..., 4:5] += reference[..., 2:3]
                tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
                tmp[..., 0:1] = (tmp[..., 0:1] * (self.obj_pts_bbox_head.pc_range[3] -
                                self.obj_pts_bbox_head.pc_range[0]) + self.obj_pts_bbox_head.pc_range[0])
                tmp[..., 1:2] = (tmp[..., 1:2] * (self.obj_pts_bbox_head.pc_range[4] -
                                self.obj_pts_bbox_head.pc_range[1]) + self.obj_pts_bbox_head.pc_range[1])
                tmp[..., 4:5] = (tmp[..., 4:5] * (self.obj_pts_bbox_head.pc_range[5] -
                                self.obj_pts_bbox_head.pc_range[2]) + self.obj_pts_bbox_head.pc_range[2])

                # TODO: check if using sigmoid
                outputs_coord = tmp
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            outputs_classes = torch.stack(outputs_classes)
            outputs_coords = torch.stack(outputs_coords)

            outs_obj = {
                'bev_embed': bev_embed,
                'all_cls_scores': outputs_classes,
                'all_bbox_preds': outputs_coords,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
            }

        bbox_list = self.pts_bbox_head.get_bboxes(outs_map, img_metas, rescale=rescale)
        if len(bbox_list[0]) == 5:
            bbox_results = [self.pred2result_dist(bboxes, scores, labels, pts, pts_cls) for bboxes, scores, labels, pts, pts_cls in bbox_list]
        else:
            bbox_results = [self.pred2result(bboxes, scores, labels, pts) for bboxes, scores, labels, pts in bbox_list]

        if "seg_preds" not in outs_obj.keys():
            bbox_list_obj = self.obj_pts_bbox_head.get_bboxes(outs_obj, img_metas, rescale=rescale)
            bbox_results_obj = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list_obj
            ]
        else: 
            raise(Exception("The BEVFormer head should only perform OD"))

        return {'pts_bbox': outs_map["bev_embed"], 'pts_bbox_obj': outs_obj["bev_embed"]}, {'pts_bbox': bbox_results, 'pts_bbox_obj': bbox_results_obj}

    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        lidar_feat = None
        if self.modality == "fusion":
            lidar_feat = self.extract_lidar_feat(points)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        new_prev_bev_dict, bbox_pts_dict= self.simple_test_pts(img_feats, lidar_feat, img_metas, prev_bev, rescale=rescale)
        return new_prev_bev_dict, [bbox_pts_dict]
