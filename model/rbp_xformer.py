from torch import nn, einsum
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.enformer_rbp import GELU, ConvBlock, Enformer, Residual, TargetLengthCrop, exponential_linspace_int, get_positional_embed, relative_shift
from einops import rearrange
from einops.layers.torch import Rearrange


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim1,
        dim2,
        *,
        num_rel_pos_features,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.,
        config=None
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads
        self.config = config

        self.to_q = nn.Linear(dim1, dim_key * heads, bias = False)
        self.to_k = nn.Linear(dim2, dim_key * heads, bias = False)
        self.to_v1 = nn.Linear(dim1, dim_value * heads, bias = False)
        self.to_v2 = nn.Linear(dim2, dim_value * heads, bias = False)

        self.to_out1 = nn.Linear(dim_value * heads, dim1)
        self.to_out2 = nn.Linear(dim_value * heads, dim1)
        # nn.init.zeros_(self.to_out1.weight)
        # nn.init.zeros_(self.to_out1.bias)
        nn.init.xavier_normal_(self.to_out1.weight)
        nn.init.xavier_normal_(self.to_out2.weight)

        nn.init.zeros_(self.to_out2.weight)
        nn.init.zeros_(self.to_out2.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts
        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, x1, x2, attn_mask=None, exchange=False):
        n1, n2, h, device = x1.shape[-2], x2.shape[-2], self.heads, x1.device

        q = self.to_q(x1)
        k = self.to_k(x2)
        v2 = self.to_v2(x2)

        q, k, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v2))

        q = q * self.scale

        content_logits = einsum('b h i d, b h j d -> b h i j', q, k)

        if self.config.cross_pos:
            positions = get_positional_embed(n1, self.num_rel_pos_features, device)
            positions = self.pos_dropout(positions)#.half()
            rel_q = self.to_rel_k(positions)

            rel_q = rearrange(rel_q, 'n (h d) -> h n d', h = h)
            rel_logits = einsum('b h i d, h j d -> b h i j', k * self.scale + self.rel_pos_bias, rel_q)
            rel_logits = relative_shift(rel_logits)
            rel_logits = rearrange(rel_logits, 'b h l2 l1 -> b h l1 l2', h = h)
            logits = content_logits + rel_logits
        else:
            logits = content_logits
        
        if attn_mask is not None:
            logits += attn_mask
        attn = logits.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out1 = einsum('b h i j, b h j d -> b h i d', attn, v2)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        if exchange:
            v1 = self.to_v1(x1)

            v1 = rearrange(v1, 'b n (h d) -> b h n d', h = h)
            attn2 = rearrange(attn, 'b h n1 n2 -> b h n2 n1')
            out2 = einsum('b h i j, b h j d -> b h i d', attn2, v1)
            out2 = rearrange(out2, 'b h n d -> b n (h d)')
            return self.to_out1(out1), self.to_out2(out2)
        return self.to_out1(out1), x2
    


class CrossFormerLayer(nn.Module):
    def __init__(self, config, first_layer=False):
        super().__init__()
        if first_layer:
            self.first = True
            self.res_proj = nn.Linear(config.esm_dim+config.cell_line_dim, config.dim, bias=False)
        else:
            self.first = False
            if config.exchange:
                self.res_proj = nn.Identity()
            else:
                self.res_proj = nn.Linear(config.esm_dim+config.cell_line_dim, config.dim, bias=False)

        dim2 = config.dim
        self.config = config
        self.LayerNormx= nn.LayerNorm(config.dim)
        self.LayerNormy= nn.LayerNorm(dim2)

        if config.front_proj:
            self.proj_x = nn.Linear(config.dim, config.dim)
            self.proj_y = nn.Linear(dim2, config.dim)

        self.attn = CrossAttention(
            config.dim,
            dim2,
            # config.dim,
            heads = config.heads,
            dim_key = config.attn_dim_key,
            dim_value = config.dim // config.heads,
            dropout = config.attn_dropout,
            pos_dropout = config.pos_dropout,
            num_rel_pos_features = config.dim // config.heads,
            config=config
        )
        self.attn_dropout = nn.Dropout(config.dropout_rate)

        self.ffnx = Residual(nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim * 2),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(config.dim * 2, config.dim),
            nn.Dropout(config.dropout_rate)
        ))

        self.ffny = Residual(nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim * 2),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(config.dim * 2, config.dim),
            nn.Dropout(config.dropout_rate)
        ))

        self.protein_proj = nn.Linear(dim2, config.dim)

    def forward(self, x, y0, attn_mask=None, exchange=False):
        if self.config.front_proj:
            x = self.proj_x(x)
        y = self.res_proj(y0)
        x1 = self.LayerNormx(x)
        y1 = self.LayerNormy(y)
        x2, y2 = self.attn(x1, y1, attn_mask, exchange)
        x3 = self.attn_dropout(x2)

        x4 = x + x3

        x5 = self.ffnx(x4)

        if exchange:
            y3 = self.attn_dropout(y2)
            if self.config.norm_res:
                y4 = y3 + y1
            else:
                y4 = y3 + y
            y5 = self.ffny(y4)
        elif self.config.linear_change_pro:
            y5 = self.protein_proj(y) + y
        else:
            y5 = y0

        return x5, y5
    
class CrossFormerLayer(nn.Module):
    def __init__(self, config, first_layer=False):
        super().__init__()
        if first_layer:
            self.first = True
            self.res_proj = nn.Linear(config.esm_dim+config.cell_line_dim, config.dim, bias=False)
        else:
            self.first = False
            if config.exchange:
                self.res_proj = nn.Identity()

            elif config.linear_change_pro:
                self.res_proj = nn.Identity()
            else:
                self.res_proj = nn.Linear(config.esm_dim+config.cell_line_dim, config.dim, bias=False)

        dim2 = config.dim
        self.config = config
        self.LayerNormx= nn.LayerNorm(config.dim)
        self.LayerNormy= nn.LayerNorm(dim2)


        self.attn = CrossAttention(
            config.dim,
            dim2,
            # config.dim,
            heads = config.heads,
            dim_key = config.attn_dim_key,
            dim_value = config.dim // config.heads,
            dropout = config.attn_dropout,
            pos_dropout = config.pos_dropout,
            num_rel_pos_features = config.dim // config.heads,
            config=config
        )
        self.attn_dropout = nn.Dropout(config.dropout_rate)

        self.ffnx = Residual(nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim * 2),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(config.dim * 2, config.dim),
            nn.Dropout(config.dropout_rate)
        ))

        self.ffny = Residual(nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim * 2),
            nn.Dropout(config.dropout_rate),
            nn.ReLU(),
            nn.Linear(config.dim * 2, config.dim),
            nn.Dropout(config.dropout_rate)
        ))

        self.protein_proj = nn.Linear(dim2, config.dim)

    def forward(self, x, y0, attn_mask=None, exchange=False):
        y = self.res_proj(y0)
        x1 = self.LayerNormx(x)
        y1 = self.LayerNormy(y)
        x2, y2 = self.attn(x1, y1, attn_mask, exchange)
        x3 = self.attn_dropout(x2)

        x4 = x + x3

        x5 = self.ffnx(x4)

        if exchange:
            y3 = self.attn_dropout(y2)
            y4 = y3 + y
            y5 = self.ffny(y4)
        else:
            y5 = y0

        return x5, y5


class CrossOutLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(config.dim * 2, config.dim)
        self.act_fn = nn.GELU()
        self.out = nn.Linear(config.dim, 1)

    def forward(self, x, y):
        b, n1, d = x.shape
        _, n2, d = y.shape

        x_expanded = x.unsqueeze(2)
        x_expanded = x_expanded.repeat(1, 1, n2, 1)
        y_expanded = y.unsqueeze(1)
        y_expanded = y_expanded.repeat(1, n1, 1, 1)

        input = torch.concat([x_expanded, y_expanded], dim=-1)

        o1 = self.linear1(input)
        o2 = self.act_fn(o1)
        o3 = self.out(o2)
        o4 = o3.squeeze(3)

        return o4
    
class CrossOutLayer_2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(config.dim * 3, config.dim)
        self.act_fn = nn.GELU()
        self.out = nn.Linear(config.dim, 1)

    def forward(self, x0, x, y):
        b, n1, d = x.shape
        _, n2, d = y.shape
        
        x0_expanded = x0.unsqueeze(2)
        x0_expanded = x0_expanded.repeat(1, 1, n2, 1)
        x_expanded = x.unsqueeze(2)
        x_expanded = x_expanded.repeat(1, 1, n2, 1)
        y_expanded = y.unsqueeze(1)
        y_expanded = y_expanded.repeat(1, n1, 1, 1)

        input = torch.concat([x0_expanded, x_expanded, y_expanded], dim=-1)

        o1 = self.linear1(input)
        o2 = self.act_fn(o1)
        o3 = self.out(o2)
        o4 = o3.squeeze(3)

        return o4
    
class OutStage1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.dim, config.dim // 2)
        self.act_fn = nn.GELU()
        self.out = nn.Linear(config.dim // 2, 1)
    
    def forward(
            self,
            x
    ):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.out(x)

        return x

class ResidualMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.dim*2, config.dim)
        self.fc2 = nn.Linear(config.dim, config.dim)
        self.fc3 = nn.Linear(config.dim, 166)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class OutStage3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.dim * 4, config.num_label)
        nn.init.zeros_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)


    def forward(self, x):
        x = self.linear1(x)
        return x
    

class RBPXFormer(nn.Module):
    def __init__(self, config, enformer_param_dir=None):
        super().__init__()
        self.config = config
        self.enformer = Enformer(config)

        self.cross_attn = CrossFormerLayer(config, first_layer=True)

        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2
        
        filter_list = exponential_linspace_int(half_dim, config.dim, num = (config.num_downsamples - 1), divisible_by = config.dim_divisible_by)
        filter_list = [half_dim, *filter_list]

        if enformer_param_dir is not None:
            self.load_enformer(enformer_param_dir, freeze=False)
 

        self.cell_line_emb_layer = nn.Embedding(num_embeddings=self.config.n_cell_line, embedding_dim=config.cell_line_dim) if config.cell_line_dim > 0 else None

        self.crop_final = TargetLengthCrop(config.target_length)
        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(config.dropout_rate / 8),
            GELU()
        )
        self.out_stage1 = OutStage1(config)
        self.out_stage2 = CrossOutLayer(config)
        self.out_stage2_2 = CrossOutLayer_2(config)
        self.out_stage3 = OutStage3(config)
        
        

    def load_enformer(self, model_dir, freeze=True):
        state_dict = torch.load(model_dir, map_location=torch.device('cpu'))
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v 
            else:
                new_state_dict[k] = v

        model_dict = self.enformer.state_dict()
        # 过滤掉形状不匹配的参数
        filtered_dict = {}
        for k, v in new_state_dict.items():
            if k in model_dict:
                if model_dict[k].size() == v.size():
                    filtered_dict[k] = v
                else:
                    print(f"跳过形状不匹配的参数: {k}，预训练形状 {v.size()}，模型要求 {model_dict[k].size()}")
            else:
                print(f"预训练参数 {k} 不在当前模型中")

        model_dict.update(filtered_dict)

        self.enformer.load_state_dict(model_dict, strict=False)
        if freeze:
            for param in self.enformer.parameters():
                param.requires_grad = False
            print("freeze RBP Former backbone.")
        print("Load RBP Former backbone parameters from {}.".format(model_dir))


    def rbp_out(
            self,
            esm2_reps,
            cell_line_tensor
    ):
        cell_line_emb = self.cell_line_emb_layer(cell_line_tensor)
        y = torch.cat([esm2_reps, cell_line_emb], dim=2)
        return y
    
    def load_enformer_only(self, model_dir, freeze=True, module_names=None):
        """
        只加载模型中指定模块的参数
        Args:
            model_dir: 模型参数文件路径
            freeze: 是否冻结加载的参数
            module_names: 要加载的模块名列表，如果为None则只加载enformer模块
        """
        state_dict = torch.load(model_dir, map_location=torch.device('cpu'))
        
        # 处理多GPU训练保存的模型
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:] if k.startswith('module.') else k: v 
                         for k, v in state_dict.items()}
        
        # 如果没有指定模块名，默认只加载enformer
        if module_names is None:
            module_names = ['enformer']
        
        # 筛选要加载的参数
        filtered_state_dict = {}
        skipped_layers = []
        shape_mismatch_layers = []
        
        # 获取模型当前的所有参数名
        model_state_dict = self.state_dict()
        
        for name, param in state_dict.items():
            # 检查是否属于要加载的模块
            should_load = any(module_name in name for module_name in module_names)
            if not should_load:
                continue
                
            # 尝试不同的参数名匹配方式
            possible_names = []
            
            # 1. 直接使用原始名称
            possible_names.append(name)
            
            # 2. 移除模块名前缀
            for module_name in module_names:
                if module_name in name:
                    possible_names.append(name.replace(f'{module_name}.', ''))
                    # 3. 对于cross_attn模块，尝试添加crossformer前缀
                    if module_name == 'cross_attn':
                        possible_names.append(f'crossformer.{name}')
                        possible_names.append(f'crossformer.{name.replace(f"{module_name}.", "")}')
                    break
            
            # 4. 尝试匹配模型中的参数名
            matched = False
            for possible_name in possible_names:
                if possible_name in model_state_dict:
                    if model_state_dict[possible_name].shape == param.shape:
                        filtered_state_dict[possible_name] = param
                        matched = True
                        break
                    else:
                        shape_mismatch_layers.append(
                            (name, param.shape, model_state_dict[possible_name].shape))
                        matched = True
                        break
            
            if not matched:
                skipped_layers.append((name, "not_in_model"))
        
        # 加载参数
        self.load_state_dict(filtered_state_dict, strict=False)
        
        # 冻结参数
        if freeze:
            for name, param in self.named_parameters():
                if any(module_name in name for module_name in module_names):
                    param.requires_grad = False
            print(f"\nFrozen parameters for modules: {module_names}")
        
        # 打印详细信息
        print(f"\nLoaded {len(filtered_state_dict)} parameters from {model_dir}")
        print("\nLoaded layers:")
        for layer_name in filtered_state_dict.keys():
            print(f"  - {layer_name}")
            
        if skipped_layers:
            print("\nSkipped layers:")
            for layer_name, reason in skipped_layers:
                print(f"  - {layer_name} (reason: {reason})")
        
        if shape_mismatch_layers:
            print("\nShape mismatch layers:")
            for layer_name, loaded_shape, required_shape in shape_mismatch_layers:
                print(f"  - {layer_name}: loaded shape {loaded_shape} vs required shape {required_shape}")
        
        return {
            'loaded_layers': list(filtered_state_dict.keys()),
            'skipped_layers': skipped_layers,
            'shape_mismatch_layers': shape_mismatch_layers
        }

    def forward(
        self,
        x,
        esm2_reps, 
        cell_line_tensor,
        enformer1_out=False,
        rbp_out=False
    ):
        if self.config.cell_line_dim > 0:
            cell_line_emb = self.cell_line_emb_layer(cell_line_tensor)
            y = torch.cat([esm2_reps, cell_line_emb], dim=2)
        else:
            y = esm2_reps

        if rbp_out:
            return esm2_reps

        if enformer1_out:
            return self.enformer(x)

        logits0, x0, x_old =  self.enformer(x, return_both=True)

        x, y = self.cross_attn(
            x0, y, exchange=True
        )
        x1 = x
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        x_old = F.dropout(x_old, p=0.2, training=self.training)
        x_final = torch.cat([x_old, x], dim=-1)
        logits1 = self.out_stage3(x_final)

        if self.config.res_out:
            return  logits1, x1, y  
        elif self.config.zero:
            return x1, y
        else:
            return self.out_stage3(x), self.out_stage1(y).squeeze(-1)

    def freeze_modules(self, module_names):
        for name, module in self.named_modules():
            if any(module_name in name for module_name in module_names):
                for param in module.parameters():
                    param.requires_grad = False
                if hasattr(module, 'eval'):
                    module.eval()
      
    def freeze_and_disable_dropout_by_name(self, module_names):
        for name, module in self.named_modules():
            if any(module_name in name for module_name in module_names):
                # 冻结参数
                for param in module.parameters():
                    param.requires_grad = False
                # 设置为评估模式，这会禁用dropout等层
                if hasattr(module, 'eval'):
                    module.eval()
                # print(f"Froze and disabled dropout for module: {name}")
    
    def unfreeze_module_by_name(self, module_names):
        for name, module in self.named_modules():
            if any(module_name in name for module_name in module_names):
                for param in module.parameters():
                    param.requires_grad = True
                if isinstance(module, nn.Dropout):
                    module.train()

    def check_module_status(self):
        status_dict = {}
        for name, module in self.named_modules():
            if len(list(module.parameters())) > 0:
                params_frozen = all(not p.requires_grad for p in module.parameters())
                status_dict[name] = {
                    'frozen': params_frozen,
                    'is_dropout': isinstance(module, nn.Dropout),
                    'training': module.training
                }
        return status_dict

    def print_module_status(self):
        status = self.check_module_status()
        for name, info in status.items():
            print(f"\nModule: {name}")
            print(f"  - Parameters frozen: {info['frozen']}")
            if info['is_dropout']:
                print(f"  - Dropout training mode: {info['training']}")
        
    def load_partial_state_dict(self, 
                              state_dict_path, 
                              target_module=None, 
                              skip_size_mismatch=True,
                              verbose=True,
                              skip_layers=None,
                              only_load_layers=None):
        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
        
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:] if k.startswith('module.') else k: v 
                         for k, v in state_dict.items()}
        
        target_model = target_module if target_module is not None else self
        model_state_dict = target_model.state_dict()
        
        filtered_state_dict = {}
        skipped_layers = []
        shape_mismatch_layers = []
        
        for name, param in state_dict.items():
            if skip_layers and any(skip in name for skip in skip_layers):
                skipped_layers.append((name, "user_specified"))
                continue
                
            if only_load_layers and not any(load in name for load in only_load_layers):
                skipped_layers.append((name, "not_in_only_load_layers"))
                continue
                
            if name not in model_state_dict:
                skipped_layers.append((name, "not_in_model"))
                continue
                
            if param.shape != model_state_dict[name].shape:
                shape_mismatch_layers.append(
                    (name, param.shape, model_state_dict[name].shape))
                if skip_size_mismatch:
                    continue
                    
            filtered_state_dict[name] = param
        
        target_model.load_state_dict(filtered_state_dict, strict=False)
        
        if verbose:
            self._print_load_info(filtered_state_dict, 
                                skipped_layers, 
                                shape_mismatch_layers)
        
        return {
            'loaded_layers': list(filtered_state_dict.keys()),
            'skipped_layers': skipped_layers,
            'shape_mismatch_layers': shape_mismatch_layers
        }

    def _print_load_info(self, filtered_state_dict, skipped_layers, shape_mismatch_layers):
        print(f"\nLoaded {len(filtered_state_dict)} layers")
        print(f"Skipped {len(skipped_layers)} layers")
        
        if skipped_layers:
            print("\nSkipped layers:")
            for layer, reason in skipped_layers:
                print(f"  - {layer} (reason: {reason})")
        
        if shape_mismatch_layers:
            print("\nShape mismatch layers:")
            for layer, old_shape, new_shape in shape_mismatch_layers:
                print(f"  - {layer}: {old_shape} vs {new_shape}")

    

