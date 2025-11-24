from optimizers.Muon_AdamW_raw import MuonWithAuxAdam
import torch

def MuonAdamW(params,lr,betas=(0.9, 0.95),weight_decay=0.01):
    #对适合与不适合Muon的参数进行过滤
    #1.获取适合muon的id
    p_id_for_muon_not_token_embeding = []
    p_id_for_muon_token_embeding = []
    found_token_embeding = False
    for i in range(len(params)):
        if torch.tensor(params[i].shape).tolist() == [126464,4096]:
            p_id_for_muon_token_embeding.append(i)

        if params[i].ndim >= 2 and params[i].shape[-1] != 1 and params[i].shape[-2] != 1 and torch.tensor(params[i].shape).tolist() != [126464,4096]:
            p_id_for_muon_not_token_embeding.append(i)
    p_id_for_muon = p_id_for_muon_not_token_embeding + p_id_for_muon_token_embeding[1:]
    p_id_for_muon.sort()
    p_id_not_for_muon = [i for i in range(len(params)) if i not in p_id_for_muon]

    p_for_muon = [params[i] for i in p_id_for_muon]#[126464,4096]是token embeding的维度
    p_not_for_muon = [params[i] for i in p_id_not_for_muon]
    assert len(p_for_muon) + len(p_not_for_muon) == len(params)
    for p in p_for_muon:
        print(f"Param shape: {p.shape}, ndim: {p.ndim}")
        assert p.ndim >= 2, f"Invalid param shape: {p.shape}"
    #nonhidden_params = [*model.parameters(), *model.parameters()]
    print("num of module suitable for muon:"+str(len(p_for_muon)))
    print("num of module not suitable for muon:"+str(len(p_not_for_muon)))
    param_groups = [
        dict(params=p_for_muon, use_muon=True,
            lr=lr, weight_decay=weight_decay),
        dict(params=p_not_for_muon, use_muon=False,#注意，词嵌入层也同样不适合用于Muon优化
            lr=lr, betas=betas, weight_decay=weight_decay),
    ]
    optimizer = MuonWithAuxAdam(param_groups)
    return optimizer



