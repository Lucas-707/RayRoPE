import torch

class CameraFrequency(torch.nn.Module):
    def __init__(
        self,
        feat_dim: int,
        pos_enc: str,
        type: str = "LDU", # none or LDU
        init: str = "rand_scale" # identity or random or rand_scale
    ):
        super().__init__()
        assert (feat_dim % 4) == 0, "feat_dim must be divisible by 4"
        self.num_freq = feat_dim // 4
        self.pos_enc = pos_enc
        self.type = type
        
        if type == "LDU":
            # Initialize LDU decomposition parameters for 4x4 matrices
            if init == "identity":
                # identity init
                self.L_params = torch.nn.Parameter(torch.zeros(self.num_freq, 6))
                self.D_params = torch.nn.Parameter(torch.zeros(self.num_freq, 4))  # log(1) = 0
                self.U_params = torch.nn.Parameter(torch.zeros(self.num_freq, 6))
            elif init == "rand_scale":
                # for each frequency, initialize D to a scaling matrix
                self.L_params = torch.nn.Parameter(torch.zeros(self.num_freq, 6))
                self.D_params = torch.nn.Parameter(torch.log(torch.rand(self.num_freq, 4) * 1.5 + 0.5))  # [0.5, 2.0]
                self.U_params = torch.nn.Parameter(torch.zeros(self.num_freq, 6))
            elif init == "random":
                self.L_params = torch.nn.Parameter(torch.randn(self.num_freq, 6))
                self.D_params = torch.nn.Parameter(torch.log(torch.rand(self.num_freq, 4) * 1.5 + 0.5))  # [0.5, 2.0]
                self.U_params = torch.nn.Parameter(torch.randn(self.num_freq, 6))

            else:
                raise ValueError(f"Unknown init type: {init}")
            

        else:
            pass
    
    def get_freq_matrices(self):
        """Reconstruct 4x4 matrices from LDU decomposition."""
        n_freq = self.num_freq
        device = self.L_params.device
        
        # Construct L matrices (lower triangular with 1s on diagonal)
        L = torch.eye(4, device=device).unsqueeze(0).repeat(n_freq, 1, 1)
        L[:, 1, 0] = self.L_params[:, 0]
        L[:, 2, 0] = self.L_params[:, 1] 
        L[:, 2, 1] = self.L_params[:, 2]
        L[:, 3, 0] = self.L_params[:, 3]
        L[:, 3, 1] = self.L_params[:, 4]
        L[:, 3, 2] = self.L_params[:, 5]
        
        # Construct D matrices (diagonal)
        D = torch.diag_embed(torch.exp(self.D_params))
        
        # Construct U matrices (upper triangular with 1s on diagonal)
        U = torch.eye(4, device=device).unsqueeze(0).repeat(n_freq, 1, 1)
        U[:, 0, 1] = self.U_params[:, 0]
        U[:, 0, 2] = self.U_params[:, 1]
        U[:, 0, 3] = self.U_params[:, 2]
        U[:, 1, 2] = self.U_params[:, 3]
        U[:, 1, 3] = self.U_params[:, 4]
        U[:, 2, 3] = self.U_params[:, 5]
        
        # Multiply L @ D @ U to get final matrices
        return torch.bmm(torch.bmm(L, D), U)
    
    def get_inverse_freq_matrices(self):
        """Efficiently compute inverse of projection matrices using LDU decomposition.
        
        For LDU decomposition, (LDU)^-1 = U^-1 @ D^-1 @ L^-1
        """
        n_freq = self.num_freq
        device = self.L_params.device
        
        # Construct L matrices (lower triangular with 1s on diagonal)
        L = torch.eye(4, device=device).unsqueeze(0).repeat(n_freq, 1, 1)
        L[:, 1, 0] = self.L_params[:, 0]
        L[:, 2, 0] = self.L_params[:, 1] 
        L[:, 2, 1] = self.L_params[:, 2]
        L[:, 3, 0] = self.L_params[:, 3]
        L[:, 3, 1] = self.L_params[:, 4]
        L[:, 3, 2] = self.L_params[:, 5]
        
        # Construct U matrices (upper triangular with 1s on diagonal)
        U = torch.eye(4, device=device).unsqueeze(0).repeat(n_freq, 1, 1)
        U[:, 0, 1] = self.U_params[:, 0]
        U[:, 0, 2] = self.U_params[:, 1]
        U[:, 0, 3] = self.U_params[:, 2]
        U[:, 1, 2] = self.U_params[:, 3]
        U[:, 1, 3] = self.U_params[:, 4]
        U[:, 2, 3] = self.U_params[:, 5]
        
        # Compute L^-1 using solve_triangular
        identity = torch.eye(4, device=device).unsqueeze(0).repeat(n_freq, 1, 1)
        L_inv = torch.linalg.solve_triangular(L, identity, upper=False, unitriangular=True)
        
        # Construct D^-1 (diagonal)
        D_inv = torch.diag_embed(torch.exp(-self.D_params))
        
        # Compute U^-1 using solve_triangular
        U_inv = torch.linalg.solve_triangular(U, identity, upper=True, unitriangular=True)
        
        # Multiply U^-1 @ D^-1 @ L^-1
        inverse_matrices = torch.bmm(torch.bmm(U_inv, D_inv), L_inv)
        
        # Verification: check that LDU @ LDU^-1 = I
        if False:  # Only verify during training to catch issues early
            original_matrices = torch.bmm(torch.bmm(L, torch.diag_embed(torch.exp(self.D_params))), U)
            product = torch.bmm(original_matrices, inverse_matrices)
            identity = torch.eye(4, device=device).unsqueeze(0).repeat(n_freq, 1, 1)
            error = torch.max(torch.abs(product - identity))
            if error > 1e-5:
                print(f"Warning: LDU inverse verification failed with max error: {error.item()}")
        
        return inverse_matrices
    
    def _plain_prope_apply_tiled_projmat(
        self,
        feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
        matrix: torch.Tensor,  # (batch, cameras, D, D)
    ) -> torch.Tensor:
        """Apply projection matrix to features."""
        # - seqlen => (cameras, patches_x * patches_y)
        # - feat_dim => (feat_dim // 4, 4)
        (batch, num_heads, seqlen, feat_dim) = feats.shape
        cameras = matrix.shape[1]
        assert seqlen > cameras and seqlen % cameras == 0
        patches = seqlen // cameras
        D = matrix.shape[-1]
        assert matrix.shape == (batch, cameras, self.num_freq, D, D) or matrix.shape == (batch, cameras, D, D)
        assert self.num_freq * D == feat_dim
        if len(matrix.shape) == 4:
            return torch.einsum(
                "bcij,bncpfj->bncpfi",
                matrix,
                feats.reshape((batch, num_heads, cameras, patches, self.num_freq, D)),
            ).reshape((batch, num_heads, seqlen, feat_dim))
        else:
            return torch.einsum(
                "bcfij,bncpfj->bncpfi",
                matrix,
                feats.reshape((batch, num_heads, cameras, patches, self.num_freq, D)),
            ).reshape((batch, num_heads, seqlen, feat_dim))

    def _plain_patch_rope_apply_tiled_projmat(
        self,
        feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
        matrix: torch.Tensor,  # (batch, seqlen, D, D)
    ) -> torch.Tensor:
        """Apply projection matrix to features."""
        # - seqlen => (cameras, patches_x * patches_y)
        # - feat_dim => (feat_dim // 4, 4)
        (batch, num_heads, seqlen, feat_dim) = feats.shape
        D = matrix.shape[-1]
        assert matrix.shape == (batch, seqlen, self.num_freq, D, D) or matrix.shape == (batch, seqlen, D, D)
        assert self.num_freq * D == feat_dim
        if len(matrix.shape) == 4:
            return torch.einsum(
                "bsij,bnsfj->bnsfi",
                matrix,
                feats.reshape((batch, num_heads, seqlen, self.num_freq, D)),
            ).reshape((batch, num_heads, seqlen, feat_dim))
        else:
            return torch.einsum(
                "bsfij,bnsfj->bnsfi",
                matrix,
                feats.reshape((batch, num_heads, seqlen, self.num_freq, D)),
            ).reshape((batch, num_heads, seqlen, feat_dim))
    
    def apply_tiled_projmat_buggy(
        self,
        feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
        matrix: torch.Tensor,  # (batch, cameras, D, D) or (batch, seqlen, D, D)
        side: str, # 'left' or 'right'
        transpose: bool = False,
        inverse: bool = False,
    ) -> torch.Tensor:
        if self.pos_enc == 'prope':
            intermediate = self._plain_prope_apply_tiled_projmat(
                feats, matrix)
        else:
            intermediate = self._plain_patch_rope_apply_tiled_projmat(
                feats, matrix) 
        # intermediate (batch, num_heads, seqlen, feat_dim // D, D)

        if self.type == "none":
            return intermediate.reshape(feats.shape)
        else:
            if inverse:
                freq_matrices = self.get_inverse_freq_matrices()
            else:
                freq_matrices = self.get_freq_matrices()  # (num_projections, 4, 4)
            if transpose:
                freq_matrices = freq_matrices.transpose(1, 2)

            if side == 'left':
                result = torch.einsum(
                    "kij,bnskj->bnski",
                    freq_matrices,
                    intermediate,
                )
            elif side == 'right':
                result = torch.einsum(
                    "bnskj,kji->bnski",
                    intermediate,
                    freq_matrices,
                )
            else:
                raise ValueError(f"Unknown side: {side}")
            
            return result.reshape(feats.shape)
        
    def apply_tiled_projmat(
        self,
        feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
        matrix: torch.Tensor,  # (batch, cameras, D, D) or (batch, seqlen, D, D)
        side: str, # 'left' or 'right'
        transpose: bool = False,
        inverse: bool = False,
    ) -> torch.Tensor:
        
        # B, C, D, _ = matrix.shape
        # matrix = matrix.view(B, C, 1, D, D).expand(-1, -1, self.num_freq, -1, -1)  # (B, C, num_freq, D, D)        
        if not self.type == 'none':
            if inverse:
                freq_matrices = self.get_inverse_freq_matrices()
            else:
                freq_matrices = self.get_freq_matrices()  # (num_projections, 4, 4)
            if transpose:
                freq_matrices = freq_matrices.transpose(1, 2)
            # freq matrices shape: (num_freq, 4, 4)
            
            if side == 'left':
                matrix = torch.einsum(
                    "fij,bcjk->bcfik",
                    freq_matrices,
                    matrix,
                )  # (B, C, num_freq, D, D)
            elif side == 'right':
                matrix = torch.einsum(
                    "bcij,fjk->bcfik",
                    matrix,
                    freq_matrices,
                )  # (B, C, num_freq, D, D)
        
        
        if self.pos_enc == 'prope':
            result = self._plain_prope_apply_tiled_projmat(
                feats, matrix)
        else:
            result = self._plain_patch_rope_apply_tiled_projmat(
                feats, matrix)
            
        assert result.shape == feats.shape
        return result
        
            


class RoPETransform(torch.nn.Module):
    def __init__(
        self,
        rope_mat_dim: int,
        num_rope_freqs: int,
        type: str = "Cayley", # none or Cayley
    ):
        super().__init__()
        self.rope_mat_dim = rope_mat_dim
        self.num_rope_freqs = num_rope_freqs
        self.transform_type = "none" if type == "none" else "Cayley"
        self.init = "randn" if "randn" in type else "identity"
        
        if self.transform_type == "Cayley":
            # Cayley parameterization: Q = (I + S)(I - S)^-1 where S is skew-symmetric
            # For an n x n skew-symmetric matrix, we need n(n-1)/2 parameters
            num_skew_params = rope_mat_dim * (rope_mat_dim - 1) // 2
            if self.init == "identity":
                self.skew_params = torch.nn.Parameter(torch.zeros(num_rope_freqs, num_skew_params))
            elif self.init == "randn":
                self.skew_params = torch.nn.Parameter(torch.randn(num_rope_freqs, num_skew_params))
            else:
                raise ValueError(f"Unknown init type: {self.init}")
        else:
            pass
    
    def get_transform_matrices(self):
        """Construct orthogonal matrices using Cayley parameterization: Q = (I - S)(I + S)^-1"""
        if self.transform_type != "Cayley":
            raise NotImplementedError(f"Transform type {self.transform_type} not implemented")
        
        device = self.skew_params.device
        n = self.rope_mat_dim
        num_freqs = self.num_rope_freqs
        
        # Construct skew-symmetric matrices S from parameters
        S = torch.zeros(num_freqs, n, n, device=device)
        
        # Fill upper triangular part with parameters
        param_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                S[:, i, j] = self.skew_params[:, param_idx]
                S[:, j, i] = -self.skew_params[:, param_idx]  # Skew-symmetric
                param_idx += 1
        
        # Construct identity matrices
        I = torch.eye(n, device=device).unsqueeze(0).repeat(num_freqs, 1, 1)
        
        # Compute Q = (I - S)(I + S)^-1
        # More efficient than computing inverse directly
        # Solve (I + S)^T @ Q^T = (I - S)^T for Q
        Q = torch.linalg.solve((I + S).transpose(-1, -2), (I - S).transpose(-1, -2)).transpose(-1, -2)

        
        return Q

    def get_inverse_transform_matrices(self):
        """Construct inverse orthogonal matrices: Q^-1 = (I + S)(I - S)^-1"""
        if self.transform_type != "Cayley":
            raise NotImplementedError(f"Transform type {self.transform_type} not implemented")

        Q_inv = self.get_transform_matrices().transpose(-1, -2)
        
        return Q_inv
    
    def apply_transform(
        self,
        rope_matrices: torch.Tensor,  # (batch, seqlen, num_freqs, D, D)
    ) -> torch.Tensor:
        if self.transform_type == "none":
            return rope_matrices
        
        U = self.get_transform_matrices()
        U_inv = self.get_inverse_transform_matrices()
      
        # Apply transform to rope matrices
        # transformed = U @ rope_matrices @ U_inv
        transformed = torch.einsum(
            "fij,bsfjk->bsfik",
            U,
            rope_matrices,
        )
        transformed = torch.einsum(
            "bsfij,fjk->bsfik",
            transformed,
            U_inv,
        )
        
        return transformed
    
    def _plain_apply_tiled_projmat(
        self,
        feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
        matrix: torch.Tensor,  # (batch, seqlen, num_freqs, D, D)
    ) -> torch.Tensor:
        """Apply projection matrix to features."""
        (batch, num_heads, seqlen, feat_dim) = feats.shape
        assert matrix.shape == (batch, seqlen, self.num_rope_freqs, self.rope_mat_dim, self.rope_mat_dim)
        assert self.rope_mat_dim * self.num_rope_freqs == feat_dim
        return torch.einsum(
            "bsfij,bnsfj->bnsfi",
            matrix,
            feats.reshape((batch, num_heads, seqlen, self.num_rope_freqs, self.rope_mat_dim)),
        ).reshape((batch, num_heads, seqlen, feat_dim))
    
    def apply_tiled_projmat_buggy(
        self,
        feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
        matrix: torch.Tensor,  # (batch, seqlen, num_freqs, D, D)
        side: str, # 'left' or 'right'
        transpose: bool = False,
        inverse: bool = False,
    ) -> torch.Tensor:
        intermediate = self._plain_apply_tiled_projmat(feats, matrix)
        # intermediate (batch, num_heads, seqlen, num_freqs, mat_dim)

        if self.type == "none":
            return intermediate.reshape(feats.shape)
        else:
            if inverse:
                freq_matrices = self.get_inverse_freq_matrices()
            else:
                freq_matrices = self.get_freq_matrices()  # (num_projections, 4, 4)
            if transpose:
                freq_matrices = freq_matrices.transpose(1, 2)

            if side == 'left':
                result = torch.einsum(
                    "fij,bnsfj->bnsfi",
                    freq_matrices,
                    intermediate,
                )
            elif side == 'right':
                result = torch.einsum(
                    "bnsfj,fji->bnsfi",
                    intermediate,
                    freq_matrices,
                )
            else:
                raise ValueError(f"Unknown side: {side}")
            
            return result.reshape(feats.shape)
        
    def apply_tiled_projmat(
        self,
        feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
        matrix: torch.Tensor,  # (batch, seqlen, num_freqs, D, D)
        side: str, # 'left' or 'right'
        transpose: bool = False,
        inverse: bool = False,
    ) -> torch.Tensor:
        

        if not self.transform_type == "none":
            if inverse:
                freq_matrices = self.get_inverse_transform_matrices()
            else:
                freq_matrices = self.get_transform_matrices()  # (num_projections, 4, 4)
            if transpose:
                freq_matrices = freq_matrices.transpose(1, 2)
            # freq matrices shape: (num_freq, D, D)


            if side == 'left':
                matrix = torch.einsum(
                    "fij,bsfjk->bsfik",
                    freq_matrices,
                    matrix,
                )  # (B, seqlen, num_freqs, D, D)
            elif side == 'right':
                matrix = torch.einsum(
                    "bsfij,fjk->bsfik",
                    matrix,
                    freq_matrices,
                )  # (B, seqlen, num_freqs, D, D)

        result = self._plain_apply_tiled_projmat(feats, matrix)
        assert result.shape == feats.shape
        return result