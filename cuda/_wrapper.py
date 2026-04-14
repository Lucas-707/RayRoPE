# cuda/_wrapper.py
import torch
from torch import Tensor
from torch.autograd import Function

from rayrope_cuda_ext import fused_rayrope_coeffs_fwd, fused_rayrope_coeffs_bwd
from .cuda_manager import GeometryKernelManager


class FusedRayRoPEFunction(Function):
    @staticmethod
    def forward(ctx,
                feats: Tensor,
                positions: Tensor,
                log_min_freqs: Tensor,
                log_max_freqs: Tensor,
                # freqs: Tensor,
                # interleaved: bool = False,
                inverse: bool = False):
        """
        feats: (batch, num_heads, seqlen, feat_dim)
        positions: (batch, num_cameras, num_patches, coord_dim)
        freqs: (num_freqs,)
        """        
        out = torch.empty_like(feats, memory_format=torch.contiguous_format)
        fused_rayrope_coeffs_fwd(
            # inputs
            feats.contiguous(),
            positions.contiguous(),
            log_min_freqs.contiguous(),
            log_max_freqs.contiguous(),
            # freqs.contiguous(), 
            # interleaved,
            inverse,
            # output
            out)
        
        ctx.save_for_backward(feats, positions, log_min_freqs, log_max_freqs)
        # ctx.interleaved = interleaved
        ctx.inverse = inverse
        
        return out

    @staticmethod
    def backward(ctx, v_out: Tensor):        
        feats, positions, log_min_freqs, log_max_freqs = ctx.saved_tensors
        # interleaved = ctx.interleaved
        inverse = ctx.inverse

        v_feats = torch.empty_like(feats, memory_format=torch.contiguous_format)
        v_positions = torch.empty_like(positions, memory_format=torch.contiguous_format)
        
        fused_rayrope_coeffs_bwd(
            # intputs
            feats.contiguous(),
            positions.contiguous(),
            log_min_freqs.contiguous(),
            log_max_freqs.contiguous(),
            #freqs.contiguous(),
            #interleaved, 
            inverse,
            # output
            # grad_output
            v_out.contiguous(),
            # grad_inputs
            v_feats,
            v_positions
        )
    
        return v_feats, v_positions, None, None, None

class FusedGeometry_KV(Function):
    @staticmethod
    def forward(ctx,
                pos_enc_type,
                batch, num_cameras, num_patches, rope_coord_dim,
                patches_x, patches_y,
                predicted_d: Tensor,
                P_inv: Tensor,
                P_mat: Tensor, 
                w2c: Tensor, 
                c2w: Tensor):
        cuda_module = GeometryKernelManager.get_kernel(pos_enc_type)
        
        # Shape: (2*B, C_q, C_kv, P, coord_dim)
        pos_KV = torch.empty(
            (2 * batch, num_cameras, num_cameras, num_patches, rope_coord_dim),
            dtype=predicted_d.dtype,
            device=predicted_d.device
        )
        
        ctx.d_shape = predicted_d.shape
        predicted_d = predicted_d.view(batch, num_cameras, num_patches, -1)
                
        cuda_module.forward(
            # inputs
            patches_x, patches_y,
            predicted_d.contiguous(),
            P_inv.contiguous(),
            P_mat.contiguous(), 
            w2c.contiguous(), 
            c2w.contiguous(),
            # output
            pos_KV.contiguous()
        )
        
        ctx.save_for_backward(predicted_d, P_inv, P_mat, w2c, c2w)
        ctx.pos_enc_type = pos_enc_type
        ctx.patches_x = patches_x
        ctx.patches_y = patches_y
        
        return pos_KV
    
    @staticmethod
    def backward(ctx, v_pos_KV):
        pos_enc_type = ctx.pos_enc_type
        
        predicted_d, P_inv, P_mat, w2c, c2w = ctx.saved_tensors
        patches_x = ctx.patches_x 
        patches_y = ctx.patches_y
        
        cuda_module = GeometryKernelManager.get_kernel(pos_enc_type)
        
        v_predicted_d = torch.empty_like(predicted_d, memory_format=torch.contiguous_format)
        
        cuda_module.backward(
            # inputs
            patches_x, patches_y,
            predicted_d.contiguous(), 
            P_inv.contiguous(), 
            P_mat.contiguous(),
            w2c.contiguous(),
            c2w.contiguous(),
            # output
            # grad_output
            v_pos_KV.contiguous(),
            # grad_input
            v_predicted_d.contiguous()
        )
        
        v_predicted_d = v_predicted_d.view(ctx.d_shape)
        
        return None,None, None, None, None, None, None, v_predicted_d, None, None, None, None

