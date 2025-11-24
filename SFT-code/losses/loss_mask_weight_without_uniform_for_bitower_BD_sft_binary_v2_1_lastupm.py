import torch 
import torch.nn.functional as F 


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

def uniform_scheduler(t):
        return 4 * 0.1 * t * (1 - t)


class MaskWithoutUniformLoss:
    def __init__(
        self,
        #ratio_dist: str,
        mask_token_id: int = 126336,
        block_size: int = 32,
        max_remask_ratio: float = 0.1, 
        upm_loss_type: str = "nll",
        upm_loss_weight: float = 0.05,
        upm_tset_weight: float = 1.0,
        upm_fset_weight: float = 1.0,
        prob_threshold:float = 0.5,
        *args, **kwargs
    ):
        #self.ratio_dist = ratio_dist
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        self.max_remask_ratio = max_remask_ratio
        self.upm_loss_type = upm_loss_type
        self.upm_loss_weight = upm_loss_weight
        self.upm_set_weights = (upm_tset_weight, upm_fset_weight)
        self.prob_threshold = prob_threshold

    def uniform_scheduler(self, t):
        return 4 * self.max_remask_ratio * t * (1 - t)

    def __call__(self, model, batch, device, precision=torch.bfloat16):

        #################################


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
            question = batch[0][i,:start_id[i]].unsqueeze(0).to(device)
            answer = batch[0][i,start_id[i]:text_length[i]].unsqueeze(0).to(device)
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
        # 只算mask_token的损失，如果要对block内全体计算loss的话需要就直接用mask_sft，不要点乘
        mask_token_in_sft = torch.mul(mask_sft, mask)
        p_mask = transform_mask(mask_token_in_sft, block_size, merge_range)
        xt = torch.where(mask ,126336,merge_qa ).long()

        timestep = p_mask
        uniform_ratio = uniform_scheduler(timestep)

        uniform_mask = (torch.rand_like(xt.float()) < uniform_ratio) & ~mask & mask_sft.bool()
        

        #####################将xt中uniform_mask位置的元素替换为除自身以及126336以外的元素
        # 1. 获取 xt 中所有不等于 126336 的元素，形成候选池
        allowed_values = xt[xt != 126336]
        # 2. 找出所有需要替换的位置的索引
        replace_indices = torch.where(uniform_mask)  # 返回一个元组，每个元素是一个维度的索引张量
        # 3. 获取需要替换的元素的原值
        original_values_to_replace = xt[replace_indices]
        # 4. 为每个需要替换的位置生成替换值（确保替换值不同于原值）
        replacement_values_list = []
        for orig_val in original_values_to_replace:
            # 从候选池中排除当前原值
            candidate_pool = allowed_values[allowed_values != orig_val]
            if len(candidate_pool) > 0:
                # 随机选择一个替换值
                idx = torch.randint(0, len(candidate_pool), (1,))
                new_val = candidate_pool[idx]
                replacement_values_list.append(new_val)
            else:
                # 如果没有候选值（极端情况），保留原值（即不替换）
                replacement_values_list.append(orig_val)
        # 5. 将替换值列表转换为张量
        if len(replacement_values_list) > 0:
            replacement_values = torch.stack(replacement_values_list).to(xt.device)
            # 6. 创建结果张量并进行替换
            xt[replace_indices] = replacement_values.squeeze()  # 确保形状匹配
        #######################
        
        
        # xt = torch.where(
        #     uniform_mask,
        #     torch.randint_like(xt, 0, 126336 - 1),
        #     xt,
        # )
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            out = model(
                input_ids=xt, 
                attention_bias=attention_masks, 
                position_ids=position_id)
            


        logits = out.logits.to(torch.float32)

        nll = F.cross_entropy(
            logits.float().view(bs * seqlen, -1), merge_qa.view(-1, ), reduction='none'
        ).view(bs, seqlen)
        # 只计算 mask token 的 loss
        nll = torch.where(mask_token_in_sft==1, nll, 0) / p_mask
        # 同时计算所有 response tokens 的 loss
        # nll = torch.where(mask_sft, nll, 0) / p_mask
        # 只计算 uniform noise tokens 的 loss
        # nll = torch.where(uniform_mask | mask_token_in_sft == 1, nll, 0) / p_mask

        # UPM Loss
        upm_logits = out.confidences.to(torch.float32)
        # 搜集各gorundtruth的预测概率
        probs = torch.gather(logits.softmax(dim = -1), dim=-1, index=merge_qa.long().unsqueeze(-1)).squeeze(-1)

        # uniform_mask, mask_token_in_sft, mask_sft
        # 将 upm_logits 按 block-size 分块
        predicted_x0 = logits.argmax(dim=-1)
        upm_block_logits = []
        block_is_correct = []
        uniform_block_mask = []
        mask_block_mask = []
        xt_belongto_norm = []
        #prob_higher_than_threshold = []
        n_block = []
        probs_target = []
        epsilon = 0.05

        for i in range(bs):
            cur_upm_logits = upm_logits[i, mask_sft[i].bool()].view(-1, block_size)
            cur_block_is_correct = (predicted_x0[i, mask_sft[i].bool()] == merge_qa[i, mask_sft[i].bool()]).view(-1, block_size)
            cur_block_belongto_norm = (xt[i, mask_sft[i].bool()] == merge_qa[i, mask_sft[i].bool()]).view(-1, block_size)
            #prob_higher_than_threshold.append((probs[i, mask_sft[i].bool()] >= 0.5).view(-1, block_size))
            probs_target.append((probs[i, mask_sft[i].bool()]).view(-1, block_size))
            xt_belongto_norm.append(cur_block_belongto_norm)
            block_is_correct.append(cur_block_is_correct)
            upm_block_logits.append(cur_upm_logits)
            uniform_block_mask.append(uniform_mask[i, mask_sft[i].bool()].view(-1, block_size))
            mask_block_mask.append(mask[i, mask_sft[i].bool()].view(-1, block_size))
            n_block.append(cur_block_belongto_norm.shape[0])

        upm_block_logits = torch.concat(upm_block_logits, dim=0) # (N, block_size)
        block_is_correct = torch.concat(block_is_correct, dim=0) # (N, block_size)
        uniform_block_mask = torch.concat(uniform_block_mask, dim=0) # (N, block_size)
        mask_block_mask = torch.concat(mask_block_mask, dim=0) # (N, block_size)
        #prob_higher_than_threshold = torch.concat(prob_higher_than_threshold, dim=0) # (N, block_size)
        xt_belongto_norm = torch.concat(xt_belongto_norm,dim=0)# (N, block_size)
        probs_target = torch.concat(probs_target,dim=0)
        #mask_block_mask[uniform_block_mask]

        #寻找label为1的token
        label_index = xt_belongto_norm*(1-epsilon) + mask_block_mask*probs_target*(1-2*epsilon)+mask_block_mask*epsilon +uniform_block_mask*epsilon
        #label_index = torch.logical_or(xt_belongto_norm, torch.mul(mask_block_mask,prob_higher_than_threshold)).float()
        
        #计算二元交叉熵损失
        upm_loss = F.binary_cross_entropy_with_logits(upm_block_logits, label_index.detach())
        # upm_loss_segments = torch.tensor_split(upm_loss, torch.cumsum(torch.tensor(n_block[:-1]), dim=0))
        # upm_loss = torch.stack([seg.mean() for seg in upm_loss_segments]).sum()
        # 原始梯度问题已通过 log(prob + eps) 方法修复 
        # torch.distributed.breakpoint()
        # normalised loss 
        nll = (nll.sum(dim=-1) / (torch.sum(mask_sft,dim=-1) + 1e-8)).sum()
        loss = nll + self.upm_loss_weight * upm_loss

        del merge_qa, attention_masks, mask_matrix
        
        return {
            'loss': loss,
            'nll': nll.detach().item(),
            'upm_loss': upm_loss.detach().item(),
        }
