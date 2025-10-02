import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
import numpy as np
from custom_cache_query import CustomCacheQuery
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def testing_proj(model,past_key_values,list_proj,ranks):
    """
    This function takes as input
    - the model, just for the config (hyperparameters etc...)
    - past_key_values: KQV cache in standard format
    - list_proj: projections for each layer, head, KQV
    - ranks: to cut the basis and get the projection on the subspace of the given shape
    It simulates the attention computation from the KQV cache given in input.
    It simulates the computation with and without applying dimension reduction and ouputs a bunch of metrics.
    It stores those metrics in lists of size L. Metrics that depend on an head are averaged accross all heads in the layer
    The list of metrics is the following:
        errorK = [] #relative frobenius norm error on K
        errorQ = [] #relative frobenius norm error on Q
        errorV = [] #relative frobenius norm error on V
        errorKQT = [] #relative frobenius norm error on KQ^T
        errorcoeff = [] #relative frobenius norm error coefficients (after the Softmax)
        errorVWO = [] #relative frobenius norm error on V W_o
        errorouput = [] #relative frobenius norm error on the ouput of the attention

    """
    
    g = model.config.num_attention_heads//model.config.num_key_value_heads#number of head groups for GQA


    errorK = [] #relative frobenius norm error on K
    errorQ = [] #relative frobenius norm error on Q
    errorV = [] #relative frobenius norm error on V
    errorKQT = [] #relative frobenius norm error on KQ^T
    errorcoeff = [] #relative frobenius norm error coefficients (after the Softmax)
    errorVWO = [] #relative frobenius norm error on V W_o
    errorouput = [] #relative frobenius norm error on the ouput of the attention

    for l in range(model.config.num_hidden_layers):
        print("layer:",l) 
        #original states
        key_states = past_key_values[l][0].to(torch.float)
        value_states = past_key_values[l][1].to(torch.float)
        query_states = past_key_values[l][2].to(torch.float)

        #to be made generic in the actual code
        _,h,T,d = query_states.shape
        scaling = d**(0.5)

        #computing projections
        projk = list_proj[l][0][:,:,:ranks[l][0]].unsqueeze(0)
        projq = list_proj[l][1][:,:,:ranks[l][0]].unsqueeze(0)
        projv = list_proj[l][2][:,:,:ranks[l][1]].unsqueeze(0)
        projw = list_proj[l][3][:,:,:ranks[l][1]].unsqueeze(0)

        #interleaving stuff that needs to be interleaved
        key_states = repeat_kv(key_states, g) #model.config.num_key_value_groups.
        value_states = repeat_kv(value_states, g)
        projk = repeat_kv(projk, g)
        projq = repeat_kv(projq, g)
        projv = repeat_kv(projv, g)
        projw = repeat_kv(projw, g)
        #actually we could project and then repeat

        #computing how well K and V are approximated
        key_states_a = key_states @ projk @ projq.transpose(2,3)
        errorK.append((torch.norm(key_states-key_states_a)/torch.norm(key_states)).item())
        
        value_states_a = value_states @ projv @ projw.transpose(2,3)
        errorV.append((torch.norm(value_states-value_states_a)/torch.norm(value_states)).item())
        
        query_states_a = query_states @ projq @ projk.transpose(2,3)
        errorQ.append((torch.norm(query_states-query_states_a)/torch.norm(query_states)).item())

        #compressed states
        query_states_c = torch.matmul(query_states,projq)
        key_states_c = torch.matmul(key_states,projk)
        value_states_c = torch.matmul(value_states,projv)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
        attn_weights_c = torch.matmul(query_states_c, key_states_c.transpose(2, 3)) * scaling
        errorKQT.append((torch.norm(attn_weights-attn_weights_c)/torch.norm(attn_weights)).item())

        attn_bias = torch.zeros(attn_weights.shape, dtype=attn_weights.dtype, device=attn_weights.device)
        temp_mask = torch.ones(attn_weights.shape, dtype=torch.bool).tril(diagonal=0).to(attn_weights.device) #sets elements on the upper triang to False
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf")) #upper triang is set to -inf, lower is still 0
        attn_bias.to(attn_weights.dtype)
        attn_weights = attn_weights + attn_bias
        attn_weights_c = attn_weights_c + attn_bias

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
        attn_weights_c = torch.nn.functional.softmax(attn_weights_c, dim=-1, dtype=torch.float32).to(query_states.dtype) 
        
        errorcoeff.append((torch.norm(attn_weights-attn_weights_c)/torch.norm(attn_weights)).item())
       
        attn_output = torch.matmul(attn_weights, value_states)# value_states is (batch, h, Tkv, d)
        attn_output_c = torch.matmul(attn_weights_c, value_states_c)

        #decompressing
        attn_output_c = torch.matmul(attn_output_c,projw.transpose(2,3))

        #so here attn_output is (batch, h, Tq, d)
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output_c = attn_output_c.transpose(1,2).contiguous()

        attn_output = attn_output.reshape(1, T, -1).contiguous()
        attn_output_c = attn_output_c.reshape(1, T, -1).contiguous()

        errorVWO.append((torch.norm((value_states_a.transpose(1,2).reshape(1, T, -1) - 
                                    value_states.transpose(1,2).reshape(1, T, -1) )
                         @ model.layers[l].self_attn.o_proj.weight.T.to(torch.float))
                         /torch.norm(value_states.transpose(1,2).reshape(1, T, -1) @
                                     model.layers[l].self_attn.o_proj.weight.T.to(torch.float))).item())

        attn_output = model.layers[l].self_attn.o_proj(attn_output.to(torch.float16))
        attn_output_c = model.layers[l].self_attn.o_proj(attn_output_c.to(torch.float16))
        
        errorouput.append((torch.norm(attn_output-attn_output_c)/torch.norm(attn_output)).item())
        
        #the average for each head is done by taking norm on the tensors
    
    return [errorK,errorQ,errorV,errorKQT,errorVWO,errorcoeff,errorouput]

def testing_proj_norope(model,past_key_values,list_proj,ranks):
    """
    Same thing than the previous function, but we project on the subspace (equivalent to compress and decompress)
    and then apply RoPE. This is what is done in the litterature to make the method RoPE compatible.
    Projections need to be learned on caches without RoPE.
    Time indices (which impact RoPE) are by default from 0 to T-1, where T is the sequence length of test samples
    """
    g = model.config.num_attention_heads//model.config.num_key_value_heads

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    errorK = []
    errorQ = []
    errorV = []
    errorKQT = []
    errorcoeff = []
    errorVWO = []
    errorouput = []

    for l in range(model.config.num_hidden_layers):
        print("layer:",l) 
        #original states
        key_states = past_key_values[l][0].to(torch.float)
        value_states = past_key_values[l][1].to(torch.float)
        query_states = past_key_values[l][2].to(torch.float)
        #assuming that cache is without ROPE

        #to be made generic in the actual code
        _,h,T,d = query_states.shape
        scaling = d**(0.5)
    
        hidden_states = torch.randn((1,T,d*h)).to(device)#just to simulate an hidden state
        position_ids = torch.arange(0,T).unsqueeze(0).to(device)
        position_embeddings = model.rotary_emb(hidden_states,position_ids) #._modules['model']
        cos, sin = position_embeddings

        #computing projections
        projk = list_proj[l][0][:,:,:ranks[l][0]].unsqueeze(0)
        projq = list_proj[l][1][:,:,:ranks[l][0]].unsqueeze(0)
        projv = list_proj[l][2][:,:,:ranks[l][1]].unsqueeze(0)
        projw = list_proj[l][3][:,:,:ranks[l][1]].unsqueeze(0)

        #interleaving stuff that needs to be interleaved
        key_states = repeat_kv(key_states, g) #model.config.num_key_value_groups.
        value_states = repeat_kv(value_states, g)
        projk = repeat_kv(projk, g)
        projq = repeat_kv(projq, g)
        projv = repeat_kv(projv, g)
        projw = repeat_kv(projw, g)

        #computing how well K and V are approximated
        key_states_a = key_states @ projk @ projq.transpose(2,3)
        value_states_a = value_states @ projv @ projw.transpose(2,3)
        query_states_a = query_states @ projq @ projk.transpose(2,3)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states_a, key_states_a = apply_rotary_pos_emb(query_states_a, key_states_a, cos, sin)
        

        #compressed states
        errorK.append((torch.norm(key_states-key_states_a)/torch.norm(key_states)).item())
        errorV.append((torch.norm(value_states-value_states_a)/torch.norm(value_states)).item())
        errorQ.append((torch.norm(query_states-query_states_a)/torch.norm(query_states)).item())

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
        attn_weights_c = torch.matmul(query_states_a, key_states_a.transpose(2, 3)) * scaling
        errorKQT.append((torch.norm(attn_weights-attn_weights_c)/torch.norm(attn_weights)).item())

        attn_bias = torch.zeros(attn_weights.shape, dtype=attn_weights.dtype, device=attn_weights.device)
        temp_mask = torch.ones(attn_weights.shape, dtype=torch.bool).tril(diagonal=0).to(attn_weights.device) #sets elements on the upper triang to False
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf")) #upper triang is set to -inf, lower is still 0
        attn_bias.to(attn_weights.dtype)
        attn_weights = attn_weights + attn_bias
        attn_weights_c = attn_weights_c + attn_bias

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
        attn_weights_c = torch.nn.functional.softmax(attn_weights_c, dim=-1, dtype=torch.float32).to(query_states.dtype) 
        
        errorcoeff.append((torch.norm(attn_weights-attn_weights_c)/torch.norm(attn_weights)).item())
       
        attn_output = torch.matmul(attn_weights, value_states)# value_states is (batch, h, Tkv, d)
        attn_output_c = torch.matmul(attn_weights_c, value_states_a)

        #so here attn_output is (batch, h, Tq, d)
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output_c = attn_output_c.transpose(1,2).contiguous()

        attn_output = attn_output.reshape(1, T, -1).contiguous()
        attn_output_c = attn_output_c.reshape(1, T, -1).contiguous()

        errorVWO.append((torch.norm((value_states_a.transpose(1,2).reshape(1, T, -1) - 
                                    value_states.transpose(1,2).reshape(1, T, -1) )
                         @ model.layers[l].self_attn.o_proj.weight.T.to(torch.float))
                         /torch.norm(value_states.transpose(1,2).reshape(1, T, -1) @
                                     model.layers[l].self_attn.o_proj.weight.T.to(torch.float))).item())
        
        attn_output = model.layers[l].self_attn.o_proj(attn_output.to(torch.float16))
        attn_output_c = model.layers[l].self_attn.o_proj(attn_output_c.to(torch.float16))

        
        errorouput.append((torch.norm(attn_output-attn_output_c)/torch.norm(attn_output)).item())
        
        #the average for each head is done quite naturally by taking norm on the tensors
    
    return [errorK,errorQ,errorV,errorKQT,errorVWO,errorcoeff,errorouput]


def validation_error(model,valdata,list_proj,ranks,rope=False):
    """
    Takes a list of validation sequences of tokens, a list of projection and some ranks.
    Produces the KQV cache for all those sequences and simulates the attention computation for those caches.
    Averages on all the sequences the metrics and returns them.
    Can be used with or without RoPE
    """
    L = model.config.num_hidden_layers
    averr = np.zeros((7,L))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    for i,input_ids in enumerate(valdata):
        print("seq:",i)
        past_key_values = CustomCacheQuery()

        with torch.no_grad():
            output = model(input_ids.to(device),past_key_values=past_key_values)
        
        if rope:
            err7 = np.array(testing_proj(model,output.past_key_values,list_proj,ranks))
        else:
            err7 = np.array(testing_proj_norope(model,output.past_key_values,list_proj,ranks))

        averr += err7

    averr /= len(valdata)
    return averr

def plotting_metrics(Oerror,Nerror,save=False,save_name=None):

    f, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(3, 2)
    f.set_figheight(15)
    f.set_figwidth(20)

    ax1.plot(range(1,33),Nerror[0],color="firebrick",marker="s",label="K cache approx error")
    ax1.plot(range(1,33),Nerror[2],color="firebrick",marker="o",label="V cache approx error")
    ax1.plot(range(1,33),Nerror[1],color="firebrick",marker="^",label="Q 'cache' approx error")
    #
    ax1.plot(range(1,33),Oerror[0],color="lightblue",marker="s",label="K cache approx error")
    ax1.plot(range(1,33),Oerror[2],color="lightblue",marker="o",label="V cache approx error")
    ax1.plot(range(1,33),Oerror[1],color="lightblue",marker="^",label="Q 'cache' approx error")
    _, y_max1 = ax1.get_ylim()
    ax1.set_ylim(0, y_max1)
    # ax1.semilogy()
    ax1.legend()

    ax3.plot(range(1,33),Nerror[3],color="firebrick",marker="s",label="K@Q.T approx error")
    ax4.plot(range(1,33),Nerror[4],color="firebrick",marker="o",label="V@WO approx error")
    #1
    ax3.plot(range(1,33),Oerror[3],color="lightblue",marker="s",label="K@Q.T approx error")
    ax4.plot(range(1,33),Oerror[4],color="lightblue",marker="o",label="V@WO approx error")
    #
    ax2.plot(range(1,33),Nerror[5],color="firebrick",marker="s",label="attn_weights approx error")
    ax2.plot(range(1,33),Oerror[5],color="lightblue",marker="s",label="attn_weights approx error")
    # ax2.semilogy()
    _, y_max2 = ax2.get_ylim()
    ax2.set_ylim(0, y_max2)
    ax2.legend()
    _, y_max3 = ax3.get_ylim()
    ax3.set_ylim(0, y_max3)
    ax3.legend()
    _, y_max4 = ax4.get_ylim()
    ax4.set_ylim(0, y_max4)
    ax4.legend()

    ax5.plot(range(1,33),Nerror[6],color="firebrick",marker="s",label="output approx error")
    #
    ax5.plot(range(1,33),Oerror[6],color="lightblue",marker="s",label="output approx error")
    _, y_max5 = ax5.get_ylim()
    ax5.set_ylim(0, y_max5)
    # ax3.semilogy()
    ax5.legend()

    if save:
        assert save_name is not None
        f.savefig(save_name) 