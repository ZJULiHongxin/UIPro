import torch
import torch.nn as nn

from einops import rearrange
from transformers.models.deformable_detr import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoder,
    DeformableDetrDecoderLayer,
    DeformableDetrDecoderOutput,
)

from .sampler import build_eos_tokens, build_pos_embeds
from .slice_logic import rearrange_image_feature, cal_num_of_slices

class DAbstractor(DeformableDetrDecoder):
    # reference: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/deformable_detr/modeling_deformable_detr.py#1279
    def __init__(
        self, 
        num_queries: int,
        embed_dim: int,
        output_hidden_size: int = 1024,
        num_eos_tokens: int = 0,
        use_posemb: bool = True,
        num_input_tokens: int = 576,
        feature_layer_index: int = -1,
        pooled_v_target: str = 'query',
        use_out_proj: bool = False,
        **igargs
    ):
        self.module_type = 'Dabs'
        config = DeformableDetrConfig(
            d_model=embed_dim, # hidden_dim
            decoder_layers=2,
            # encoder_attention_heads=8,
            # decoder_attention_heads=8,
            use_pretrained_backbone=False,  # False: from scratch
            initializer_range=0.02,  # initialization std for eos tokens
            disable_custom_kernels=False,  # use custom cuda kernel or pytorch implementation
            # below is for ablation
            num_feature_levels=1,
            feature_layer_index=-1,  # vision feature layer index; -1: last layer
            manual_init_refPoints=True,
            learnable_mRP=True,
            num_queries=num_queries,  # required
            num_query_tokens=num_queries)

        super().__init__(config)

        self.num_queries = config.num_queries
        self.num_input_tokens = num_input_tokens

        self.num_feature_levels = config.num_feature_levels
        self.isMs = self.num_feature_levels > 1

        self.layers = nn.ModuleList(
            [DeformableDetrDecoderLayer(config) for _ in range(config.decoder_layers)]
        )

        # define level_emb layer
        if self.isMs:  # for multi-scale features
            assert config.num_feature_levels == len(config.feature_layer_index)
            self.level_emb = nn.Parameter(
                torch.Tensor(1, config.num_feature_levels, 1, config.d_model)
            )
            nn.init.normal_(self.level_emb)  # same initialize with the original implementation

        # initialize the query embeddings as pooled visual feature map
        self.pooled_v_target = pooled_v_target
        if self.pooled_v_target != "none":
            tgt_hw = int(config.num_queries**0.5)
            self.downsampler = nn.AdaptiveAvgPool2d((tgt_hw, tgt_hw))
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)
        else:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model * 2)

        # define reference points
        # manual initialization + make them as learable parameters
        valid_ratios_q, spatial_shapes_q, _ = self._prepare_ddetr_inputs(1, num_input_tokens, 1)
        reference_points = self._get_query_reference_points(spatial_shapes_q, valid_ratios_q)
        self.reference_points = nn.Parameter(reference_points)

        # think tokens
        self.eos_tokens = build_eos_tokens(num_eos_tokens, config.d_model)

        # pos emb
        self.v_pos_emb = build_pos_embeds(num_input_tokens, config.d_model, use_posemb=use_posemb)

        # token projector
        if use_out_proj and output_hidden_size != config.d_model:
            self.output_proj = nn.Linear(config.d_model, output_hidden_size)
        else:
            self.output_proj = nn.Identity()

    def _get_query_reference_points(self, spatial_shapes, valid_ratios):
        """
        Get reference points for each feature map. Used in decoder.
        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        steps = int(self.num_queries**0.5)
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, steps, dtype=torch.float32),
                torch.linspace(0.5, width - 0.5, steps, dtype=torch.float32),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points.squeeze(2)

    def _forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        intermediate = ()
        intermediate_reference_points = ()

        for _, decoder_layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError("Reference points' last dimension must be of size 2")
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if False: #self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    position_embeddings,
                    reference_points_input,
                    spatial_shapes,
                    level_start_index,
                    encoder_hidden_states, # vit feat.
                    encoder_attention_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    reference_points=reference_points_input.to(dtype=hidden_states.dtype),
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return DeformableDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _process_v_features(self, visual_feat):
        # visual_feat: [B, len, dim] or [B, lvls, len, dim]

        if self.isMs:
            visual_feats = []
            for level in range(self.num_feature_levels):
                visual_feats.append(visual_feat[:, level])
            visual_feat = torch.stack(visual_feats, 1)

            # add pos emb [1, len, dim]
            if self.v_pos_emb is not None:
                visual_feat = visual_feat + self.v_pos_emb.unsqueeze(1)

            # add lvl emb [1, lvls, 1, dim]
            visual_feat = visual_feat + self.level_emb
            visual_feat = visual_feat.flatten(1, 2)  # [B, lvls, v_len, dim] -> [B, lvls*v_len, dim]
        else:
            if self.v_pos_emb is not None:
                visual_feat = visual_feat + self.v_pos_emb

        return visual_feat

    def _convert_dtype_device(self, tgt_feat, dtype=None, device=None):
        # tgt_feat: target tensor to be converted
        _dtype = tgt_feat.dtype if dtype is None else dtype
        _device = tgt_feat.device if device is None else device

        tgt_feat = tgt_feat.type(_dtype).to(_device)

        return tgt_feat

    def _prepare_ddetr_inputs(self, batch_size, seq_len, lvls, dtype=None, device=None):
        # assume there are no paddings in a feature map
        valid_ratios = torch.ones(batch_size, lvls, 2)

        # assume all feature maps have the same sequence length (i.e., the same shape)
        spatial_shapes = torch.tensor([int(seq_len**0.5), int(seq_len**0.5)]).repeat(lvls, 1)
        level_start_index = torch.arange(0, seq_len * lvls, seq_len)

        if dtype is not None and device is not None:
            valid_ratios = self._convert_dtype_device(valid_ratios, dtype=dtype, device=device)
            spatial_shapes = self._convert_dtype_device(
                spatial_shapes, dtype=torch.long, device=device
            )
            level_start_index = self._convert_dtype_device(
                level_start_index, dtype=torch.long, device=device
            )

        return valid_ratios, spatial_shapes, level_start_index

    def _make_pooled_queries(self, visual_feat):
        assert (
            self.num_feature_levels == 1
        )  # currently do not support multi-scale features for the v-pooled Q

        batch_size, seq_len, h_dim = visual_feat.shape
        query_embeds = self.query_position_embeddings.weight
        if self.pooled_v_target != "none":
            hw_v = int(seq_len**0.5)
            hw_q = int(self.num_queries**0.5)
            visual_feat = rearrange(visual_feat, "b (h w) d -> b d h w", h=hw_v, w=hw_v)
            if self.pooled_v_target == "tgt":
                query_embed = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                target = self.downsampler(visual_feat)
                target = rearrange(target, "b d h w -> b (h w) d", h=hw_q, w=hw_q)
            else:
                target = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                query_embed = self.downsampler(visual_feat)
                query_embed = rearrange(query_embed, "b d h w -> b (h w) d", h=hw_q, w=hw_q)
        else:
            query_embed, target = torch.split(query_embeds, h_dim, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
            target = target.unsqueeze(0).expand(batch_size, -1, -1)

        return query_embed, target

    def forward(self, visual_feat):
        """
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
        """
        # Old:deformable attention only supports fp32 | NOTE: pytorch2.2 已经支持grid_sample_2d_cuda使用bf16
        # original_dtype = visual_feat.type()
        #visual_feat = visual_feat.type(torch.cuda.FloatTensor)
        visual_feat = self._process_v_features(visual_feat)

        batch_size, seq_len, h_dim = visual_feat.shape
        seq_len /= self.num_feature_levels

        query_embed, target = self._make_pooled_queries(visual_feat)
        
        #target = target.to(torch.float32)

        reference_points = self.reference_points.expand(batch_size, -1, -1)

        valid_ratios, spatial_shapes, level_start_index = self._prepare_ddetr_inputs(
            batch_size, seq_len, self.num_feature_levels, visual_feat.dtype, visual_feat.device
        )

        decoder_outputs_dict = self._forward(
            inputs_embeds=target,
            position_embeddings=query_embed,
            encoder_hidden_states=visual_feat,
            valid_ratios=valid_ratios,
            reference_points=reference_points,
            return_dict=True,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        decoder_outputs = decoder_outputs_dict.last_hidden_state

        if self.eos_tokens is not None:
            decoder_outputs = torch.cat(
                [decoder_outputs, self.eos_tokens.expand(batch_size, -1, -1)], dim=1
            )

        decoder_outputs = self.output_proj(decoder_outputs)
        # decoder_outputs = decoder_outputs.type(original_dtype)

        return decoder_outputs


class SpatialAwareCompressor(DAbstractor):
    def __init__(
        self, 
        num_queries: int,
        embed_dim: int,
        output_hidden_size: int = 1024,
        num_eos_tokens: int = 0,
        use_posemb: bool = True,
        num_input_tokens: int = 576,
        feature_layer_index: int = -1,
        pooled_v_target: str = 'query',
        use_out_proj: bool = False,
        **igargs
    ):
        super().__init__(num_queries, embed_dim, output_hidden_size, num_eos_tokens, use_posemb, num_input_tokens, feature_layer_index, pooled_v_target, use_out_proj, **igargs)
        
        # ground query
        # self.ground_query = nn.Parameter(torch.randn(1, config.d_model))
        self.ground_position_embeddings = nn.Parameter(torch.randn(1, embed_dim))
        
        
    def _get_grounding_query_embeds(self, image_features, reference_points):
        batch_size, height, width, d_model = image_features.shape
        h_index = torch.floor(reference_points[:, :, 0] * height).squeeze().long() # batch_size, 1
        w_index = torch.floor(reference_points[:, :, 1] * width).squeeze().long() # batch_size, 1
        query_embeds = image_features[torch.arange(batch_size), h_index, w_index] # batch_size, d_model
        query_embeds = query_embeds.unsqueeze(1) # batch_size, 1, d_model
        
        return query_embeds
    
    def _get_grounding_feature(
        self,
        visual_feat: torch.Tensor,
        reference_points: torch.Tensor,
    ):
        """
        Args:
            visual_feat torch.FloatTensor of shape (batch_size, h, w, hidden_size)
            reference_points torch.FloatTensor of shape (batch_size, 1, 2):
                format (h, w) in [0, 1]
        """
        batch_size, h, w, hidden_size = visual_feat.shape
        
        # no additional position embedding due to the real shape of image is different
        visual_feat = self.input_proj[0](visual_feat)
        
        query_embeds = self._get_grounding_query_embeds(visual_feat, reference_points)
        query_position_embeds = self.ground_position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        query_embeds = query_embeds + query_position_embeds # batch_size, 1, d_model
        
        spatial_shapes = torch.tensor([h, w]).unsqueeze(0) # 1, 2
        valid_ratios = torch.ones(batch_size, 1, 2) # batch_size, 1, 2
        
        spatial_shapes = self._convert_dtype_device(spatial_shapes, dtype=torch.long, device=visual_feat.device)
        valid_ratios = self._convert_dtype_device(valid_ratios, dtype=visual_feat.dtype, device=visual_feat.device)
        
        visual_feat = rearrange(visual_feat, 'b h w c -> b (h w) c') # batch_size, h*w, d_model
        
        output_dict = self._forward(
            inputs_embeds=query_embeds,
            position_embeddings=query_position_embeds,
            encoder_hidden_states=visual_feat,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            return_dict=True,
            level_start_index=0,
        )
        
        output = output_dict.last_hidden_state
        
        output = self.output_proj(output)
        
        return output
        

    def get_grounding_feature(
        self,
        visual_feat: torch.Tensor,
        reference_points: torch.Tensor,
        original_image_width: int,
        original_image_height: int,
    ):
        """
        cannot process batched inputs, due to the real shape of image is different
        Args:
            visual_feat torch.FloatTensor of shape (num_slices, seq_len, hidden_size)
            reference_points torch.FloatTensor of shape (num_points, 2):
                format (w, h) in [0, 1000]
        """
        # normalize the reference points
        reference_points = reference_points / 1000.
        
        best_w, best_h = cal_num_of_slices(original_image_width, original_image_height)
        
        visual_feat_to_process = []
        reference_points_to_process = []
        
        for point in reference_points:
            w, h = point # notice here, the width is the first dimension. Belows, the h_index and w_index are swapped
            h_index = min(int(h * best_h), best_h - 1)
            w_index = min(int(w * best_w), best_w - 1)
            slice_index = h_index * best_w + w_index
            
            local_feat = visual_feat[slice_index]
            visual_feat_to_process.append(local_feat)
            
            local_h = h - h_index / best_h
            local_w = w - w_index / best_w
            reference_points_to_process.append(torch.tensor([local_h, local_w]))
            
            global_feat = visual_feat[-1]
            visual_feat_to_process.append(global_feat)
            reference_points_to_process.append(torch.tensor([h, w]))
        
        visual_feat_to_process = torch.stack(visual_feat_to_process, 0)
        reference_points_to_process = torch.stack(reference_points_to_process, 0).unsqueeze(1) # num_points, 1, 2
        reference_points_to_process = self._convert_dtype_device(
            reference_points_to_process, dtype=visual_feat.dtype, device=visual_feat.device
        )
        
        visual_feat_to_process = rearrange_image_feature(visual_feat_to_process, original_image_width, original_image_height)
        
        output = self._get_grounding_feature(visual_feat_to_process, reference_points_to_process)
        
        # the output is the concatenation of local and global features
        # we need to split them
        local_output = output[::2]
        global_output = output[1::2]
        
        # combine the local and global features
        if self.config.g_combine == "add":
            output = local_output + global_output
        else:
            raise NotImplementedError(f"combine method {self.config.g_combine} is not implemented")
        
        return output