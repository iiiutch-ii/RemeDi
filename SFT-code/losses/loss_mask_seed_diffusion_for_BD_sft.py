import torch 
import torch.nn.functional as F 
import copy
import random
import math

def transform_mask(mask_sft, block_size,merge_range):
        n_batch, seq_length = mask_sft.shape
        p_mask = mask_sft.clone().float()  # 避免修改原张量
        eps = 0.0001

        for i in range(n_batch):
            row = mask_sft[i]
            # 找到连续1的起始和结束位置
            start, end = merge_range[i][2][0],  merge_range[i][2][1]

            # 分块处理
            for j in range(start, end + 1, block_size):
                block_end = min(j + block_size, end + 1)
                block = row[j:block_end]
                r = block.float().mean()  # 计算1的比例
                p_mask[i, j:block_end] = r
        
        p_mask = (1-eps)*p_mask + eps
        return p_mask


def sample_mask_ratio(size, device):
    ratio_dist = "uniform"
    d_mean =0 
    d_std =1
    t_range = (0, 1) 
    max_uniform_ratio = 0.2
    if ratio_dist == 'uniform':
        ratio = torch.rand((size,), device=device)
    elif ratio_dist == 'logitnormal':
        z = torch.randn((size,), device=device) * d_std + d_mean
        ratio = torch.sigmoid(z)
    else:
        raise NotImplementedError 
    ratio = ratio * (t_range[1] - t_range[0]) + t_range[0]
    return ratio


def delete_tokens_with_levenshtein1(A, m1, m2, endtoken=None,device=torch.device("cpu")):
    """
    Args:
        A: 1D Tensor of shape [m]
        m1, m2: Delete tokens in range [m1, m2] (0-based, inclusive)
        K: Number of tokens to delete (Levenshtein distance)
        endtoken: Optional padding token if deletion goes beyond m2
    Returns:
        Modified Tensor with K tokens deleted in [m1, m2]
    """
    A = A.clone()  # Avoid modifying original tensor
    m = A.shape[0]
    # Ensure [m1, m2] is valid
    assert 0 <= m1 <= m2 <= m, "Invalid range [m1, m2]"
    #assert (m2 - m1 + 1) >= K, "Cannot delete K tokens in [m1, m2] (not enough tokens)"
    
    # Randomly select K unique positions in [m1, m2]
    delete_positions = sorted(random.sample(range(m1, m2), 1))
    
    # Perform deletion and shift left
    mask = torch.ones(m, dtype=torch.bool,device=device)
    mask[delete_positions] = False
    A_modified = A[mask]
    
    # If deletion goes beyond m2, pad with endtoken (optional)
    if endtoken is not None and len(A_modified) < m:
        padding_length = m - len(A_modified)
        if A.shape[0] - m2 <= 1:
            A_modified = torch.cat([A_modified, torch.full((padding_length,), endtoken, dtype=A.dtype).to(A_modified.device)])
        else:
            A_modified = torch.cat([A_modified, A[m2:m2+1]])
    return A_modified
'''
A = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12])
A_modified_0 = delete_tokens_with_levenshtein1(A, 4, 12, endtoken=126081,device=torch.device("cpu"))
A_modified_1 = delete_tokens_with_levenshtein1(A_modified_0, 4, 12, endtoken=126081,device=torch.device("cpu"))
A_modified_2 = delete_tokens_with_levenshtein1(A_modified_1, 4, 12, endtoken=126081,device=torch.device("cpu"))
'''
def insert_tokens_with_levenshtein1(A, m1, m2, endtoken=None,device=torch.device("cpu")):
    """
    Args:
        A: 1D Tensor of shape [m]
        m1, m2: Delete tokens in range [m1, m2] (0-based, inclusive)
        K: Number of tokens to delete (Levenshtein distance)
        endtoken: Optional padding token if deletion goes beyond m2
    Returns:
        Modified Tensor with K tokens deleted in [m1, m2]
    """
    A_modified = A.clone()  # Avoid modifying original tensor
    m = A.shape[0]
    # Ensure [m1, m2] is valid
    assert 0 <= m1 <= m2 <= m, "Invalid range [m1, m2]"
    #assert (m2 - m1 + 1) >= K, "Cannot delete K tokens in [m1, m2] (not enough tokens)"
    
    # Randomly select K unique positions in [m1, m2]
    insert_positions = random.sample(range(m1, m2), 1)
    
    # Perform deletion and shift left
    A_modified[m1:m2] = torch.cat([A[m1:insert_positions[0]],torch.randint(low=0, high=126336-1, size=(1,),device=device),A[insert_positions[0]:m2-1]],0)

    return A_modified
'''
A = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12])
A_modified_0 = insert_tokens_with_levenshtein1(A, 4, 12, endtoken=126081,device=torch.device("cpu"))
A_modified_1 = insert_tokens_with_levenshtein1(A_modified_0, 4, 12, endtoken=126081,device=torch.device("cpu"))
A_modified_2 = insert_tokens_with_levenshtein1(A_modified_1, 4, 12, endtoken=126081,device=torch.device("cpu"))
'''
def subsitute_tokens_with_levenshtein1(A, m1, m2, endtoken=None,device=torch.device("cpu")):
    """
    Args:
        A: 1D Tensor of shape [m]
        m1, m2: Delete tokens in range [m1, m2] (0-based, inclusive)
        K: Number of tokens to delete (Levenshtein distance)
        endtoken: Optional padding token if deletion goes beyond m2
    Returns:
        Modified Tensor with K tokens deleted in [m1, m2]
    """
    A_modified = A.clone()  # Avoid modifying original tensor
    m = A.shape[0]
    # Ensure [m1, m2] is valid
    assert 0 <= m1 <= m2 <= m, "Invalid range [m1, m2]"
    #assert (m2 - m1 + 1) >= K, "Cannot delete K tokens in [m1, m2] (not enough tokens)"
    
    # Randomly select K unique positions in [m1, m2]
    subsitute_positions = random.sample(range(m1, m2), 1)
    
    # Perform deletion and shift left
    A_modified[subsitute_positions[0]] = torch.randint(low=0, high=126336-1, size=(1,),device=device)

    return A_modified
'''
A = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12])
A_modified_0 = subsitute_tokens_with_levenshtein1(A, 4, 12, endtoken=126081,device=torch.device("cpu"))
A_modified_1 = subsitute_tokens_with_levenshtein1(A_modified_0, 4, 12, endtoken=126081,device=torch.device("cpu"))
A_modified_2 = subsitute_tokens_with_levenshtein1(A_modified_1, 4, 12, endtoken=126081,device=torch.device("cpu"))
'''

@torch.no_grad()
def edit(token_tensor,merge_range,block_size=32,L_dist_prop_threshold=0.1):
    n = 0
    for j in range(len(merge_range)):
        n_block = (merge_range[j][2][1] - merge_range[j][2][0])//block_size
        for k in range(n_block):
            L_dist = math.floor(block_size*random.random()*L_dist_prop_threshold)
            edit_type = random.choices(["deletion","insertion","subsitution"],k=L_dist)
            for m in range(L_dist):
                n+=1
                # print(n)
                if edit_type[m] == "deletion":
                    token_tensor[j,:] = delete_tokens_with_levenshtein1(token_tensor[j,:], merge_range[j][2][0]+block_size*k, merge_range[j][2][0]+block_size*(k+1), endtoken=126081,device=token_tensor.device)

                if edit_type[m] == "insertion":
                    token_tensor[j,:] = insert_tokens_with_levenshtein1(token_tensor[j,:], merge_range[j][2][0]+block_size*k, merge_range[j][2][0]+block_size*(k+1), endtoken=126081,device=token_tensor.device)

                if edit_type[m] == "subsitution":
                    token_tensor[j,:] = subsitute_tokens_with_levenshtein1(token_tensor[j,:], merge_range[j][2][0]+block_size*k, merge_range[j][2][0]+block_size*(k+1), endtoken=126081,device=token_tensor.device)
    return token_tensor
'''
token_tensor_0 = edit(token_tensor=merge_qa[0:2,:],merge_range=merge_range[0:2],block_size=32,L_dist_prop_threshold=0.1)
token_tensor=merge_qa[0:2,:]
merge_range=merge_range[0:2]
block_size=32
L_dist_prop_threshold=0.1
'''

class Seed_Diffusion_Loss:
    def __init__(
        self,
        #ratio_dist: str,
        mask_token_id: int = 126336,
        d_mean: float = 0.0,
        d_std: float = 1.0,
        t_min: float = 0.0, 
        t_max: float = 1.0,
        max_uniform_ratio: float = 0.2,
        block_size: int = 32
    ):
        #self.ratio_dist = ratio_dist
        self.mask_token_id = mask_token_id
        self.d_mean = d_mean 
        self.d_std = d_std
        self.t_range = (t_min, t_max) 
        self.max_uniform_ratio = max_uniform_ratio
        self.block_size = block_size
    


    def __call__(self, model, batch, device, precision=torch.bfloat16):
        block_size = self.block_size
        #首先构建sft的损失函数掩码矩阵
        x0 = batch[0].to(device)
        #x0 = torch.cat([x0,torch.ones_like(x0[:,:1])*126081],dim=-1)
        bs, seqlen = x0.size()
        #把block给padding了
        start_id = torch.tensor(batch[1],device=device)
        text_length = torch.tensor(batch[2],device=device)
        #num_padding = block_size - (text_length - start_id)%block_size
        merge_qa = []
        merge_range = []
        for i in range(bs):
            question = batch[0][i,:start_id[i]].unsqueeze(0)
            answer = batch[0][i,start_id[i]:text_length[i]].unsqueeze(0)
            if ((text_length[i] - start_id[i])%block_size).item() != 0:
                num_padding = block_size - (text_length[i] - start_id[i])%block_size
                answer = torch.cat([answer,torch.ones([1,num_padding],device=device)*126081],dim=-1)
            question_range = [0,question.shape[1]]
            answer_range = [question.shape[1],question.shape[1] + answer.shape[1]]
            mask_answer_range = [question.shape[1] + answer.shape[1],question.shape[1] + 2*answer.shape[1]]
            merge_range.append([question_range,answer_range,mask_answer_range])
            merge_qa.append(torch.cat([question,answer,answer],-1))

        #计算position_id
        position_id = []
        len_max = max([x.shape[1] for x in merge_qa])
        for i in range(bs):
            res = len_max - merge_qa[i].shape[1]
            if res >= 0:#理论上会有一个等于0
                pos_id = list(range(merge_range[i][1][1])) + list(range(merge_range[i][1][0],merge_range[i][1][1])) + [0 for a in range(res)]
                position_id.append(pos_id)
        position_id = torch.tensor(position_id,device=device)

        #把总体padding了
        for i in range(bs):
            res = len_max - merge_qa[i].shape[1]
            if res >= 0:
                merge_qa[i] = torch.cat([merge_qa[i],torch.ones([1,res],device=device)*126081],-1)
                #print(merge_qa[i].shape)
        merge_qa = torch.stack(merge_qa,dim = 0).squeeze(1).long()

        #对计算损失函数时需要掩码的部分构建mask
        bs, seqlen = merge_qa.size()
        mask_start_id = torch.tensor([merge_range[i][2][0] for i in range(bs)],device=device)
        mask_end_id = torch.tensor([merge_range[i][2][1] for i in range(bs)],device=device)
        mask_sft = (torch.arange(0, seqlen, device=device).unsqueeze(dim=0) < mask_start_id.unsqueeze(dim=1)) | (torch.arange(0, seqlen, device=device).unsqueeze(dim=0) >= mask_end_id.unsqueeze(dim=1))
        mask_sft = torch.where(mask_sft, 0, 1)#mask为布尔张量，为true则为mask，为False则为原来的x0

        #对attention中的mask进行计算
        bs = len(merge_range)
        # attention_mask_template = torch.ones([seqlen,seqlen],device=device)
        # attention_mask_template = torch.tril(attention_mask_template)
        #对每个句子计算attention_mask
        attention_masks = []
        for i in range(bs):
            #对block整体进行掩码
            n_block = int((merge_range[i][1][1] - merge_range[i][1][0])/block_size)*2 + 1 + 1#一份question，一份padding_out_of_block
            attention_mask = torch.ones([n_block,n_block],device=device)
            attention_mask = torch.tril(attention_mask)
            #block外的padding不需要考虑，需要进行掩码
            attention_mask[-1,:] = 0
            attention_mask[:,-1] = 0
            #存在掩码的block需要进行注意力权重之间的掩码
            n_block_response = int((merge_range[i][1][1] - merge_range[i][1][0])/block_size)
            for j in range(n_block_response):
                attention_mask[j+n_block_response+1,(j+1):(j+1+n_block_response)] = 0

            #构建每个block的重复次数
            repeat_time = [merge_range[i][0][1] - merge_range[i][0][0]] + [block_size for i in range(int((merge_range[i][1][1] - merge_range[i][1][0])/block_size)*2)] + [len_max - merge_range[i][2][1]]
            repeat_time = torch.tensor(repeat_time,device=device)
        
            attention_mask = torch.repeat_interleave(attention_mask, repeat_time, dim=1)
            attention_mask = torch.repeat_interleave(attention_mask, repeat_time, dim=0)
            # if attention_mask.shape[0] != len_max:
            #     torch.distributed.breakpoint()
            assert attention_mask.shape[0] == len_max
            attention_masks.append(attention_mask)

        attention_masks = torch.stack(attention_masks)
        attention_masks = attention_masks.unsqueeze(1)
        attention_masks = attention_masks == 1
        #构建文本的掩码概率矩阵
        mask_matrix = []
        for i in range(bs):
            #对block整体进行掩码
            n_block = int((merge_range[i][1][1] - merge_range[i][1][0])/block_size)*2 + 1 + 1#一份question，一份padding_out_of_block
            n_block_response = int((merge_range[i][1][1] - merge_range[i][1][0])/block_size)
            random_count = torch.rand([n_block],device=device)
            random_count[0:(1+n_block_response)] = 1
            repeat_time = [merge_range[i][0][1] - merge_range[i][0][0]] + [block_size for i in range(int((merge_range[i][1][1] - merge_range[i][1][0])/block_size)*2)] + [len_max - merge_range[i][2][1]]
            repeat_time = torch.tensor(repeat_time,device=device)
            random_count = torch.repeat_interleave(random_count, repeat_time, dim=0)
            mask_matrix.append(random_count)
        mask = torch.stack(mask_matrix,0)
        mask = mask < torch.rand([bs,seqlen],device=device)
        #对merge_qa进行edit
        xt = edit(merge_qa,merge_range,block_size=self.block_size,L_dist_prop_threshold=0.1)
        #对merge_qa进行掩码
        xt = torch.where( mask ,
                          self.mask_token_id,
                          merge_qa ).long()

        with torch.autocast(device_type='cuda', dtype=precision, enabled=True):
            logits = model(input_ids=xt,attention_bias=attention_masks,position_ids=position_id).logits


        #只算mask_token的损失，如果要对block内全体计算loss的话需要就直接用mask_sft，不要点乘
        mask_token_in_sft = torch.mul(mask_sft,mask)
        p_mask = transform_mask(mask_token_in_sft,block_size,merge_range)
        
        #禁止模型预测mask_token
        with torch.no_grad():
            # 找到 mask 为 1 的位置
            mask_indices = mask.unsqueeze(-1).expand_as(logits)
            # 直接在这些位置的 mask_token_id 上减去一个大数
            logits[mask_indices & (torch.arange(logits.size(-1), device=logits.device) == self.mask_token_id)] -= 1e8
        
        nll = F.cross_entropy(
            logits.float().view(bs * seqlen, -1), merge_qa.view(-1, ), reduction='none'
        ).view(bs, seqlen)
        #计算mask损失
        nll_mask = (torch.where(mask_token_in_sft==1,nll,0)/p_mask).sum(dim=-1) / (torch.sum(mask_sft,dim=-1) + 0.00000000001)
        #计算重构损失
        nll_recon = torch.where(mask_sft==1,nll,0).sum(dim=-1) / (torch.sum(mask_sft,dim=-1) + 0.00000000001)
        
        del merge_qa, attention_masks, mask_matrix
        
        #return nll_mask + nll_recon

        return {
            'loss': nll_mask + nll_recon,
            'nll': nll_mask.detach().item(),
            'upm_loss': nll_recon.detach().item(),
        }



