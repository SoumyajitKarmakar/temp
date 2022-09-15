import torch 
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self , input_dim , hidden_dim , output_dim , p):
        super().__init__()
        self.fc1 = nn.Linear(input_dim , hidden_dim )
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim , output_dim)
        self.drop = nn.Dropout(p)

    def forward(self , x):
        x = self.fc1(x) #(n_samples , n_patches + 1,hidden_dim)
        x = self.act1(x) #(n_samples , n_patches + 1, hidden_dim)
        x = self.fc2(x) # (n_samples , n_patches + 1 , output_dim)
        x = self.drop(x) #(n_samples , n_patches + 1, output_dim)
        return x


class Attention(nn.Module):
    def __init__(self , input_dim , n_head = 12 , qkv_bias = True , attn_prob = 0. , proj_prob = 0.):
        super().__init__()
        self.input_dim = input_dim 
        self.n_head = n_head
        self.head_dim = input_dim // n_head
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(input_dim , input_dim * 3 , bias = qkv_bias)
        self.attn_drop = nn.Dropout(p = attn_prob)
        self.proj = nn.Linear(input_dim , input_dim)
        self.proj_drop = nn.Dropout(p = proj_prob)
    

    def forward(self , x):
        n_samples , n_tokens , input_dim = x.shape
        qkv = self.qkv(x) # (n_samples , n_patches + 1, embed_dim * 3)
        qkv = qkv.reshape(n_samples, n_tokens , 3 , self.n_head , self.head_dim) #(n_samples , n_patches + 1, 3,  n_head , head_dim)
        qkv = qkv.permute(2 , 0 , 3 , 1 , 4)  #(3 , n_samples , n_head , n_patches +1 , head_dim)
        q , k , v = qkv[0] , qkv[1] , qkv[2]  #(n_samples , n_head , n_patches + 1, head_dim)
        k_t = k.transpose(2 , 3) #(n_samples , n_head , head_dim , n_patches + 1)
        dp = (q @ k_t ) * self.scale #(n_samples , n_head , n_patches + 1 , n_patches + 1)
        attn_matrix = dp.softmax(dim = -1) #(n_samples , n_head , n_patches + 1 , n_patches + 1)
        attn = self.attn_drop(attn_matrix) #(n_samples , n_head  , n_patches + 1, n_patches + 1)
        weighted_avg = attn @ v #(N_samples , n_head , n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1 , 2) #(n_samples , n_patches +1 m n_head , head_dim)
        weighted_avg = weighted_avg.flatten(2) #(n_samples , n_patches + 1, embed_dim)
        x = self.proj(weighted_avg) #(n_samples , n_patches + 1 , embed_dim )
        output = self.proj_drop(x)
        return output



class Block(nn.Module):
    def __init__(self , dim , n_head , mlp_ratio =4.0 , qkv_bias = True ,p = 0. , attn_p = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim ,eps = 1e-6)
        self.attn = Attention(dim , n_head , qkv_bias, attn_p , p)
        self.norm2 = nn.LayerNorm(dim , eps = 1e-6)
        self.mlp = MLP(dim , int(dim * mlp_ratio) , dim  , p)

    def forward(self , x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VIT(nn.Module):
    def __init__(self , embed_dim = 192 , depth = 3 , n_heads = 12 , mlp_ratio = 4. , qkv_bias = True , p = 0. , attn_p = 0.):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1 ,1 , embed_dim))
        self.blocks  = nn.ModuleList(
            [Block(embed_dim , n_heads , mlp_ratio , qkv_bias , p , attn_p) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim , eps =1e-6)
    

    def forward(self , x):
        
        n_samples = x.shape[0]
        cls_token = self.cls_token.expand(n_samples , - 1, -1) #(n_samples , 1 , embed_dim)
        x = torch.cat((cls_token , x) , dim = 1) #(n_samples , 1 + n_patches, embed_dim)
        for blocks in self.blocks:
            x = blocks(x)    #(n_samples, n_patches + 1, embed_dim)
        x = self.norm(x) #(n_samples, n_patches + 1, embed_dim)
        cls_token_final = x[: ,0] #Just the class tokens (n_samples ,1 , embed_dim)
        return cls_token_final

