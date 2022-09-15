import torch

class PatchDis:
    def __init__(self ,aggregation_module ,k , device="cuda"):
        self.index_matrix = torch.arange(0,196).view(14 , 14).tolist()
        self.cosine = torch.nn.CosineSimilarity(dim=-1)
        self.k = k
        self.pdist = torch.nn.PairwiseDistance(p=2)
        self.aggregation_module = aggregation_module
        self.neighbour_dict = {}
        for i in range(0 , 14):
            for j in range(0 , 14):
                ans = self.get_neighbours(i , j)
                self.neighbour_dict[self.index_matrix[i][j]] = torch.tensor(ans).to("cuda")

    def batched_index_select(self , input, dim, index):
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)    

    def get_patch_loss(self , patch_embed):
        loss = 0.0
        batch_size = patch_embed.shape[0]
        for index in range(0 , 196):
            neighbours = self.neighbour_dict[index]
            neighbour_embedding = torch.index_select(patch_embed , 1 , neighbours)
            current_patch_embedding = patch_embed[: , index , :].unsqueeze(1)
            output = self.cosine(neighbour_embedding, current_patch_embedding)
            indx_ = torch.argsort(output , descending=True)
            topk = indx_[: , :self.k]
        
            topkembed = self.batched_index_select(neighbour_embedding , 1 , topk)
            aggregated_result = self.aggregation_module(topkembed)
            single_patch_loss = self.get_single_patch_loss(current_patch_embedding , aggregated_result)
            loss += single_patch_loss
        return loss / (196 * batch_size)    

    def get_single_patch_loss(self , current_embedding , aggregated_result):
        current_embedding = torch.squeeze(current_embedding , 1)
        output = self.pdist(current_embedding , aggregated_result)
        return output.sum()    

    def get_neighbours(self , i , j):
        arr = self.index_matrix
        n = len(arr)
        m = len(arr[0])        
        v = []
        if (self.isValidPos(i - 1, j - 1, n, m)):
            v.append(arr[i - 1][j - 1])
        if (self.isValidPos(i - 1, j, n, m)):
            v.append(arr[i - 1][j])
        if (self.isValidPos(i - 1, j + 1, n, m)):
            v.append(arr[i - 1][j + 1])
        if (self.isValidPos(i, j - 1, n, m)):
            v.append(arr[i][j - 1])
        if (self.isValidPos(i, j + 1, n, m)):
            v.append(arr[i][j + 1])
        if (self.isValidPos(i + 1, j - 1, n, m)):
            v.append(arr[i + 1][j - 1])
        if (self.isValidPos(i + 1, j, n, m)):
            v.append(arr[i + 1][j])
        if (self.isValidPos(i + 1, j + 1, n, m)):
            v.append(arr[i + 1][j + 1])
        return v
    
    def isValidPos(self ,i, j, n, m):
        if (i < 0 or j < 0 or i > n - 1 or j > m - 1):
            return 0
        return 1
