import torch
import torch.nn as nn
from transformers import PreTrainedModel, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from smt_model.configuration_smt import SMTConfig
from smt_model.architectures.smt_arch import SMTOutput

class Qwen2_5_VLWrapper(PreTrainedModel):
    config_class = SMTConfig

    def __init__(self, config: SMTConfig):
        super().__init__(config)
        self.config = config
        
        self.w2i = config.w2i
        self.i2w = config.i2w
        self.maxlen = int(config.maxlen)
        self.padding_token = config.padding_token
        
        # Determine special token IDs
        self.bos_id = self.w2i.get('<bos>', 0)
        self.eos_id = self.w2i.get('<eos>', 1)
        
        # We append vision tokens dynamically to the end of the vocabulary
        self.vision_start_id = len(self.w2i)
        self.image_pad_id = len(self.w2i) + 1
        self.vision_end_id = len(self.w2i) + 2
        
        total_vocab_size = len(self.w2i) + 3

        # Configure Qwen2.5-VL using tutor's requested hyperparameters
        qwen_cfg = Qwen2_5_VLConfig(
            text_config=dict(
                intermediate_size=256,
                hidden_size=256,
                num_attention_heads=4,
                num_hidden_layers=config.num_dec_layers if config.num_dec_layers else 8,
                num_key_value_heads=4,
                vocab_size=total_vocab_size,
                bos_token_id=self.bos_id,
                eos_token_id=self.eos_id,
                rope_scaling={"type": "mrope", "mrope_section": [8, 12, 12]},
            ), 
            vision_config=dict(
                depth=8,
                hidden_size=256,
                intermediate_size=256,
                out_hidden_size=256,
                num_heads=4,
                in_channels=1, # Configured for grayscale natively
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2, # Fixed as per vision config typicals
                tokens_per_second=4,
                window_size=112,
                fullatt_block_indexes=tuple(range(8)),
                initializer_range=0.02
            ),
            image_token_id=self.image_pad_id,
            vision_start_token_id=self.vision_start_id,
            vision_end_token_id=self.vision_end_id
        )
        
        self.qwen_model = Qwen2_5_VLForConditionalGeneration(qwen_cfg)
        
        # SMT images might naturally be 1-channel, ensure everything aligns
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token)

    def _prepare_inputs(self, encoder_input, decoder_input):
        batch_size, _, H, W = encoder_input.shape
        device = encoder_input.device
        
        # Enforce dimensions to be multiples of 32. 
        # Qwen2.5-VL relies on patch_size=16 and spatial_merge_size=2, requiring H and W to be divisible by 32.
        pad_h = (32 - (H % 32)) % 32
        pad_w = (32 - (W % 32)) % 32
        
        if pad_h > 0 or pad_w > 0:
            encoder_input = torch.nn.functional.pad(encoder_input, (0, pad_w, 0, pad_h), value=1.0)
            H += pad_h
            W += pad_w
            
        # The Qwen2.5-VL vision module expects `pixel_values` as a flattened 2D tensor of patches
        # with shape `[total_patches, channel * 2 * patch_size * patch_size]`.
        # The exact patching boundaries are defined by `image_grid_thw`.
        
        # Calculate grid size
        grid_t = 1
        grid_h = H // 16
        grid_w = W // 16
        
        # Provide grid_thw, which is (B, 3) 
        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * batch_size, device=device)
        
        # Dynamically tile the input to satisfy `temporal_patch_size=2` required by the vision config.
        # The simplest approach is expanding the temporal dimension from 1 to 2.
        
        c = encoder_input.shape[1] # 1
        # To match temporal_patch_size=2, we expand time dimension to 2
        vid_input = encoder_input.unsqueeze(2).expand(-1, -1, 2, -1, -1) # [B, C, 2, H, W]
        
        # We need to extract patches of size (2, 16, 16).
        # We can use unfold:
        patches = vid_input.unfold(3, 16, 16).unfold(4, 16, 16) # [B, C, 2, H//16, W//16, 16, 16]
        patches = patches.permute(0, 3, 4, 1, 2, 5, 6).reshape(batch_size, grid_h * grid_w, c * 2 * 16 * 16)
        # Flatten patches across the batch as `image_grid_thw` separates items implicitly.
        pixel_values = patches.reshape(-1, c * 2 * 16 * 16)
        
        # Now construct input_ids
        num_image_tokens = (grid_h // 2) * (grid_w // 2) # Due to spatial_merge_size=2
        
        new_decoder_inputs = []
        mm_token_type_ids = []
        for b_i in range(batch_size):
            di = decoder_input[b_i]
            # di usually starts with <bos>. Let's separate it.
            if len(di) > 0 and di[0].item() == self.bos_id:
                start_tok = di[0:1]
                rest_tok = di[1:]
            else:
                start_tok = torch.tensor([], dtype=torch.long, device=device)
                rest_tok = di
                
            img_seq = [self.vision_start_id] + [self.image_pad_id] * num_image_tokens + [self.vision_end_id]
            img_tensor = torch.tensor(img_seq, dtype=torch.long, device=device)
            
            # Combine
            cmb_di = torch.cat([start_tok, img_tensor, rest_tok])
            new_decoder_inputs.append(cmb_di)
            
            # Interleaved sequences use MM Token type IDs of 1 to identify image patches, else 0.
            cmb_mm = torch.zeros_like(cmb_di)
            start_idx = len(start_tok) + 1 # offset past vision_start_id
            cmb_mm[start_idx:start_idx + num_image_tokens] = 1
            mm_token_type_ids.append(cmb_mm)
            
        input_ids = torch.stack(new_decoder_inputs)
        attention_mask = (input_ids != self.padding_token).long()
        mm_token_type_ids = torch.stack(mm_token_type_ids)
        
        return input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw

    def forward(self, encoder_input, decoder_input, labels=None):
        input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw = self._prepare_inputs(encoder_input, decoder_input)
        
        outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        
        logits = outputs.logits
        
        # Isolate text logits by slicing off the visual tokens block.
        # `input_ids` layout: [<bos>, <vision_start>, <image_pad>..., <vision_end>, <text...>]
        # `labels` target layout: [<text...>]
        # Therefore, logits predicting the text portion start exactly past the visual sequence offset.

        batch_size, seqlen_with_img, vocab_size = logits.shape
        b_idx, seqlen_orig = decoder_input.shape
        
        visual_block_len = input_ids.shape[1] - seqlen_orig
        
        # Extract logits specific directly to the textual prediction tasks.
        text_logits = logits[:, visual_block_len:, :]
        
        res_output = SMTOutput(
            logits=text_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=None
        )

        if labels is not None:
            # Calculate loss (CrossEntropy requires shape B, V, L)
            res_output.loss = self.loss(text_logits.permute(0,2,1).contiguous(), labels)
            
        return res_output

    @torch.no_grad
    def predict(self, input, convert_to_str=False, return_weights=False):
        batch_size = input.size(0)
        device = input.device
        
        # Start state with just <bos>
        bos_di = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        
        # Prepare full input format with visual tokens!
        input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw = self._prepare_inputs(input, bos_di)
        
        has_eos = torch.zeros(batch_size, dtype=torch.bool, device=device)
        eos_id = self.eos_id
        
        past_key_values = None
        predicted_sequence = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        
        outputs = None
        
        # Fast autoregressive decoding
        for i in range(self.maxlen - 1):
            outputs = self.qwen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=return_weights
            )
            
            past_key_values = outputs.past_key_values
            
            # Prediction
            next_token_logits = outputs.logits[:, -1, :]
            predicted_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            predicted_sequence = torch.cat([predicted_sequence, predicted_tokens], dim=1)
            
            has_eos |= (predicted_tokens.squeeze(1) == eos_id)
            if has_eos.all():
                break
                
            # For next iter, we only feed the new token
            input_ids = predicted_tokens
            attention_mask = torch.cat([attention_mask, torch.ones_like(predicted_tokens)], dim=1)
            # Empty visual things for next iter
            pixel_values = None
            image_grid_thw = None
            
        text_sequences = []
        for b_idx in range(batch_size):
            seq = []
            for token_id in predicted_sequence[b_idx, 1:]:
                token_val = str(token_id.item()) if convert_to_str else token_id.item()
                token_str = self.i2w.get(token_val, "")
                if token_str == '<eos>':
                    break
                seq.append(token_str)
            text_sequences.append(seq)
            
        # Dummy wrapper
        decoder_output = SMTOutput(
            logits=outputs.logits if outputs else None,
            hidden_states=outputs.hidden_states if outputs else None,
            attentions=outputs.attentions if outputs else None,
            cross_attentions=None
        )
        
        return text_sequences, decoder_output
