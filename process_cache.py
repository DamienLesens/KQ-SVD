from custom_cache_query import CustomCacheQuery
import torch
import os


def computing_cache(model,expname,list_seq,num_seq):
    """
    Applies the model on the first num_seq sequences in list_seq. 
    It stores K,Q,V cache, so it needs to be used with a model that can handle CustomCacheQuery.
    The function then stacks the cache tensors for each sequence along the time dimension.
    If GQA, query cache is averaged inside each head group
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    L = model.config.num_hidden_layers
    h = model.config.num_attention_heads
    hkv = model.config.num_key_value_heads

    # assert isinstance(model,LlamaModelQuery)
    
    for j in range(num_seq):
        print("seq: ",j)

        past_key_values = CustomCacheQuery()

        with torch.no_grad():
            input = list_seq[j].long()#this is very important, max seq lenght is only 4096 for Llama2-7b
            outputs = model(input.to(device), past_key_values=past_key_values)
            # Explicitly delete the model output to free memory
            del outputs,input
            torch.cuda.empty_cache()  # Clears any freed memory
        
        print("end model")

        for l in range(L):
            # Extract full tensors for the layer once
            K_layer = past_key_values[l][0][0]  # shape: (hkv, seq_len, head_dim)
            V_layer = past_key_values[l][1][0]  # shape: (hkv, seq_len, head_dim)
            Q_layer = past_key_values[l][2][0]  # shape: (h, seq_len, head_dim)

            if hkv<h:#if GQA stacking
                h,T,d = Q_layer.shape
                Q_layer = Q_layer.view(hkv, -1, T, d)
                Q_layer = Q_layer.reshape(hkv, -1, d)

            #now Q_layer has shape (hkv, seq_len, head_dim)

            for kind, tensor in zip(['k', 'v', 'q'], [K_layer, V_layer, Q_layer]):
                path = os.path.join( expname, kind, str(l))
                os.makedirs(path, exist_ok=True)
                torch.save(tensor, os.path.join(path, f"{j}.pt"))

        del past_key_values
        torch.cuda.empty_cache()

def compute_proj_SVD_mem(model,srcname,num_seq,resname,save=True):
    """
    This function computes SVDs on the K and V cache.
    For each layer, it stacks the first num_seq tensors it finds in the layer folder along the time dimension
    It then performs an SVD
    It stores both the right singular vectors and the spectrum.
    file is:
        -speck.pt: spectrum of K
        -specv.pt: spectrum of V
        -projk.pt: "down projector" (or basis) for K
        -projq.pt: "down projector" (or basis) for Q, same basis than K
        -projv.pt: "down projector" (or basis) for V
        -projw.pt: "down projector" (or basis) for W_O, same basis than V
    all down projectors have shape (d,"r"), "r" is d in memory, so you have to select the first r COLUMNS to get the best subspace of size r
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    L = model.config.num_hidden_layers
    h = model.config.num_key_value_heads

    for l in range(L):
        print(l)

        list_blockk = []
        list_blockv = []

        pathk = srcname+"/k/"+str(l)
        pathv = srcname+"/v/"+str(l)

        for j in range(num_seq):
            tk = torch.load(pathk+"/"+ str(j)+".pt")
            list_blockk.append(tk)#shape (h,T,d)
            tv = torch.load(pathv+"/"+ str(j)+".pt")
            list_blockv.append(tv)

        bigk= torch.cat(list_blockk,dim=1)#cat along the time dimension, which is the SECOND one
        bigv= torch.cat(list_blockv,dim=1)

        del list_blockk
        del list_blockv
        torch.cuda.empty_cache()

        #doing the svd
        bigk = bigk.to(torch.float).to(device)#(h,Thuge,d)
        bigv = bigv.to(torch.float).to(device)

        list_Vk = []
        list_Vv = []
        list_sk = []
        list_sv = []
        
        for i in range(h):
            print("     ",i)
            Uk,sk,Vk = torch.linalg.svd(bigk[i],full_matrices=False)#Vk is ("r",d), we transpose later
            Uv,sv,Vv = torch.linalg.svd(bigv[i],full_matrices=False)

            del Uk,Uv
            torch.cuda.empty_cache()
            list_Vk.append(Vk)
            list_Vv.append(Vv)
            list_sk.append(sk)
            list_sv.append(sv)
        
        del bigk,bigv
        torch.cuda.empty_cache()

        Vk = torch.stack(list_Vk,dim=0)
        Vv = torch.stack(list_Vv,dim=0)
        sk = torch.stack(list_sk,dim=0)
        sv = torch.stack(list_sv,dim=0)

        if save:
            pathbase = os.path.join(resname, str(l))
            os.makedirs(pathbase,exist_ok=True)
            torch.save(sk,os.path.join(pathbase,"speck.pt"))
            torch.save(sv,os.path.join(pathbase,"specv.pt"))

            torch.save(Vk.transpose(1,2),os.path.join(pathbase,"projk.pt"))
            torch.save(Vk.transpose(1,2),os.path.join(pathbase,"projq.pt"))
            torch.save(Vv.transpose(1,2),os.path.join(pathbase,"projv.pt"))
            torch.save(Vv.transpose(1,2),os.path.join(pathbase,"projw.pt"))

        del sk,sv,Vk,Vv
        torch.cuda.empty_cache()

def compute_proj_SVD_Eigen_mem(model_config,srcname,num_seq,resname,alpha=1,save=True):
    """
    This function works the same than compute_proj_SVD_mem.
    The only difference is that it uses query cache: Key and query cache are stacked along the time dimension before doing an SVD
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    L = model_config.num_hidden_layers
    h = model_config.num_key_value_heads

    for l in range(L):
        print(l)

        list_blockk = []
        list_blockv = []
        list_blockq = []

        pathk = srcname+"/k/"+str(l)
        pathv = srcname+"/v/"+str(l)
        pathq = srcname+"/q/"+str(l)

        for j in range(num_seq):
            tk = torch.load(pathk+"/"+ str(j)+".pt")
            tk.mul_(alpha)
            list_blockk.append(tk)
            tv = torch.load(pathv+"/"+ str(j)+".pt")
            list_blockv.append(tv)
            tq = torch.load(pathq+"/"+ str(j)+".pt")
            tq.div_(alpha)
            list_blockq.append(tq)
        
        del tk,tv,tq
        torch.cuda.empty_cache()

        bigk= torch.cat(list_blockk,dim=1)
        bigv= torch.cat(list_blockv,dim=1)
        bigq= torch.cat(list_blockq,dim=1)#q cache is already averaged in memory

        del list_blockk,list_blockv,list_blockq
        torch.cuda.empty_cache()

        #doing the svd
        bigk = bigk.to(torch.float).to(device)#(h,Thuge,d)
        bigv = bigv.to(torch.float).to(device)
        bigq = bigq.to(torch.float).to(device)
        print(bigk.shape,bigq.shape)
        

        bigk.mul_(alpha)
        bigq.div_(alpha)

        #Eigen specific part
        bigk = torch.cat([bigk,bigq],dim=1)
        del bigq
        torch.cuda.empty_cache()

        list_Vk = []
        list_Vv = []
        list_sk = []
        list_sv = []
        
        for i in range(h):
            print("     ",i)
            Uk,sk,Vk = torch.linalg.svd(bigk[i],full_matrices=False)#Vk is ("r",d), we transpose later
            Uv,sv,Vv = torch.linalg.svd(bigv[i],full_matrices=False)

            del Uk,Uv
            torch.cuda.empty_cache()
            list_Vk.append(Vk)
            list_Vv.append(Vv)
            list_sk.append(sk)
            list_sv.append(sv)
        
        del bigk,bigv
        torch.cuda.empty_cache()

        Vk = torch.stack(list_Vk,dim=0)
        Vv = torch.stack(list_Vv,dim=0)
        sk = torch.stack(list_sk,dim=0)
        sv = torch.stack(list_sv,dim=0)

        if save:
            pathbase = os.path.join(resname, str(l))
            os.makedirs(pathbase,exist_ok=True)
            torch.save(sk,os.path.join(pathbase,"speck.pt"))
            torch.save(sv,os.path.join(pathbase,"specv.pt"))

            torch.save(Vk.transpose(1,2),os.path.join(pathbase,"projk.pt"))
            torch.save(Vk.transpose(1,2),os.path.join(pathbase,"projq.pt"))
            torch.save(Vv.transpose(1,2),os.path.join(pathbase,"projv.pt"))
            torch.save(Vv.transpose(1,2),os.path.join(pathbase,"projw.pt"))

        del sk,sv,Vk,Vv
        torch.cuda.empty_cache()

def compute_proj_KQT_mem(model,srcname,num_seq,resname,save=True):

    """
    This function compute projection by doing an SVD of KQ^T
    It forms K,Q,V calibration tensors like compute_proj_SVD_mem or compute_proj_SVD_Eigen_mem.
    Saving format is the same than for previous functions
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    L = model.config.num_hidden_layers
    h = model.config.num_attention_heads#we don't need it because the query cache is already averaged in memory
    hkv = model.config.num_key_value_heads
    D = model.config.hidden_size #full hidden dimension D=h*d
    d = model.config.head_dim



    for l in range(L):
        print(l)

        list_blockk = []
        list_blockv = []
        list_blockq = []

        pathk = srcname+"/k/"+str(l)
        pathv = srcname+"/v/"+str(l)
        pathq = srcname+"/q/"+str(l)

        for j in range(num_seq):
            tk = torch.load(pathk+"/"+ str(j)+".pt")
            list_blockk.append(tk)
            tv = torch.load(pathv+"/"+ str(j)+".pt")
            list_blockv.append(tv)
            tq = torch.load(pathq+"/"+ str(j)+".pt")
            list_blockq.append(tq)
        
        del tk,tv,tq
        torch.cuda.empty_cache()

        bigk= torch.cat(list_blockk,dim=1)
        bigv= torch.cat(list_blockv,dim=1)
        bigq= torch.cat(list_blockq,dim=1)#q cache is already averaged in memory

        del list_blockk,list_blockv,list_blockq
        torch.cuda.empty_cache()

        #doing the svd
        bigk = bigk.to(torch.float).to(device)#(h,Thuge,d)
        bigv = bigv.to(torch.float).to(device)
        bigq = bigq.to(torch.float).to(device)

        print(bigk.shape,bigv.shape,bigq.shape)

        #preprocessing WO
        WOm = model.layers[l].self_attn.o_proj.weight.T.detach().to(torch.float)#shape (h*d,D)
        WO = WOm.reshape(h,d,D).transpose(1,2)
        if hkv<h:
            WO = WO.view(hkv, -1, D, d).mean(dim=1)


        list_projk = []
        list_projq = []
        list_projv = []
        list_projw = []
        list_sk = []
        list_sv = []
        #batching SVDs does not work because we are allocating too much mem
        for i in range(hkv):
            print("     ",i)
            Ki = bigk[i]
            Qi = bigq[i]
            
            QK,RK = torch.linalg.qr(Ki)
            QQ,RQ = torch.linalg.qr(Qi)

            Us,S,Vs = torch.linalg.svd(RK@RQ.T,full_matrices=False)
            U = QK @ Us
            V = Vs @ QQ.T
            list_sk.append(S)
            del Us,Vs,QK,QQ,RK,RQ
            torch.cuda.empty_cache()

            Sigmam1k = torch.diag(1/S)
            projk = Qi.T @ V.T @ Sigmam1k
            projq = (U.T @ Ki).T#TODO: ? do a square root to get a really sym stuff ?

            del Qi,Ki,U,V,S
            torch.cuda.empty_cache()


            Vi = bigv[i]
            WOi = WO[i]

            QV,RV = torch.linalg.qr(Vi)
            QW,RW = torch.linalg.qr(WOi)

            Us,S,Vs = torch.linalg.svd(RV@RW.T,full_matrices=False)
            U = QV @ Us
            V = Vs @ QW.T
            list_sv.append(S)
            del Us,Vs,QV,QW,RV,RW
            torch.cuda.empty_cache()

            Sigmam1v = torch.diag(1/S)
            projv = WOi.T @ V.T @ Sigmam1v
            projw = (U.T @ Vi).T

            del WOi,Vi,U,V,S
            torch.cuda.empty_cache()

            
            list_projk.append(projk)
            list_projq.append(projq)
            print("diff to id KQ:",torch.norm(torch.eye(d).to(projk.device)-projk@(projq.T)))
            list_projv.append(projv)
            list_projw.append(projw)
            print("diff to id VW:",torch.norm(torch.eye(d).to(projv.device)-projv@(projw.T)))
            
        
        del bigk,bigv,bigq
        torch.cuda.empty_cache()

        projk = torch.stack(list_projk,dim=0)
        projq = torch.stack(list_projq,dim=0)
        projv = torch.stack(list_projv,dim=0)
        projw = torch.stack(list_projw,dim=0)
        sk = torch.stack(list_sk,dim=0)
        sv = torch.stack(list_sv,dim=0)

        del list_projk,list_projv,list_projq,list_projw
        torch.cuda.empty_cache()

        if save:
            print("saving")
            pathbase = os.path.join(resname, str(l))
            os.makedirs(pathbase,exist_ok=True)
            torch.save(sk,os.path.join(pathbase,"speck.pt"))
            torch.save(sv,os.path.join(pathbase,"specv.pt"))

            torch.save(projk,os.path.join(pathbase,"projk.pt"))
            torch.save(projq,os.path.join(pathbase,"projq.pt"))
            torch.save(projv,os.path.join(pathbase,"projv.pt"))
            torch.save(projw,os.path.join(pathbase,"projw.pt"))

        del projk,projq,projv,projw,sk,sv
        torch.cuda.empty_cache()


def cache_right_singular_basis_mem(model, srcname,num_seq,resname,save=True):
    """
    Computes for each layer, head
    \Sigma_K, V_K, \Sigma_Q, V_Q, \Sigma_V, V_V, \Sigma_W, V_W
    Be aware that torch.linalg.svd return U,S,V^T directly, such that U @ torch.diag(S) @ V is the original matrix
    I store V, not V^T (just an arbitrary convention)
    These are stored in a folder, a subfolder for each layer, subfolder for each head ?
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    L = model.config.num_hidden_layers
    h = model.config.num_attention_heads#we don't need it because the query cache is already averaged in memory
    hkv = model.config.num_key_value_heads
    D = model.config.hidden_size #full hidden dimension D=h*d
    d = model.config.head_dim

    for l in range(L):
        print(l)

        list_blockk = []
        list_blockv = []
        list_blockq = []

        pathk = srcname+"/k/"+str(l)
        pathv = srcname+"/v/"+str(l)
        pathq = srcname+"/q/"+str(l)

        for j in range(num_seq):
            tk = torch.load(pathk+"/"+ str(j)+".pt")
            list_blockk.append(tk)
            tv = torch.load(pathv+"/"+ str(j)+".pt")
            list_blockv.append(tv)
            tq = torch.load(pathq+"/"+ str(j)+".pt")
            list_blockq.append(tq)
        
        del tk,tv,tq
        torch.cuda.empty_cache()

        bigk= torch.cat(list_blockk,dim=1)
        bigv= torch.cat(list_blockv,dim=1)
        bigq= torch.cat(list_blockq,dim=1)#q cache is already averaged in memory

        del list_blockk,list_blockv,list_blockq
        torch.cuda.empty_cache()

        #doing the svd
        bigk = bigk.to(torch.float).to(device)#(h,Thuge,d)
        bigv = bigv.to(torch.float).to(device)
        bigq = bigq.to(torch.float).to(device)

        print(bigk.shape,bigv.shape,bigq.shape)

        #preprocessing WO
        WOm = model.layers[l].self_attn.o_proj.weight.T.detach().to(torch.float)#shape (h*d,D)
        WO = WOm.reshape(h,d,D).transpose(1,2)
        if hkv<h:
            WO = WO.view(hkv, -1, D, d).mean(dim=1)
        
        #batching SVDs does not work because we are allocating too much mem
        for i in range(hkv):
            print("     ",i)
            pathbase = os.path.join(resname, str(l), str(i))
            os.makedirs(pathbase)

            Ki = bigk[i]
            Qi = bigq[i]
            Vi = bigv[i]
            WOi = WO[i]
            
            UK,SK,VK = torch.linalg.svd(Ki)
            UQ,SQ,VQ = torch.linalg.svd(Qi)
            UV,SV,VV = torch.linalg.svd(Vi)
            UW,SW,VW = torch.linalg.svd(WOi)

            del UK,UQ,UV,UW
            torch.cuda.empty_cache()
            
            torch.save(SK,os.path.join(pathbase,"sk.pt"))
            torch.save(VK.T,os.path.join(pathbase,"vk.pt"))
            torch.save(SQ,os.path.join(pathbase,"sk.pt"))
            torch.save(VQ.T,os.path.join(pathbase,"vk.pt"))
            torch.save(SV,os.path.join(pathbase,"sv.pt"))
            torch.save(VV.T,os.path.join(pathbase,"vv.pt"))
            torch.save(SW,os.path.join(pathbase,"sw.pt"))
            torch.save(VW.T,os.path.join(pathbase,"vw.pt"))

            del SK,SQ,SV,SW,VK,VQ,VV,VW
            torch.cuda.empty_cache()
            
        
        del bigk,bigv,bigq
        torch.cuda.empty_cache()

    return

def compute_proj_test_mem(model,srcname,num_seq,resname,r=110,save=True):

    """
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    L = model.config.num_hidden_layers
    h = model.config.num_attention_heads#we don't need it because the query cache is already averaged in memory
    hkv = model.config.num_key_value_heads
    D = model.config.hidden_size #full hidden dimension D=h*d
    d = model.config.head_dim



    for l in range(L):
        print(l)

        list_blockk = []
        list_blockv = []
        list_blockq = []

        pathk = srcname+"/k/"+str(l)
        pathv = srcname+"/v/"+str(l)
        pathq = srcname+"/q/"+str(l)

        for j in range(num_seq):
            tk = torch.load(pathk+"/"+ str(j)+".pt")
            list_blockk.append(tk)
            tv = torch.load(pathv+"/"+ str(j)+".pt")
            list_blockv.append(tv)
            tq = torch.load(pathq+"/"+ str(j)+".pt")
            list_blockq.append(tq)
        
        del tk,tv,tq
        torch.cuda.empty_cache()

        bigk= torch.cat(list_blockk,dim=1)
        bigv= torch.cat(list_blockv,dim=1)
        bigq= torch.cat(list_blockq,dim=1)#q cache is already averaged in memory

        del list_blockk,list_blockv,list_blockq
        torch.cuda.empty_cache()

        #doing the svd
        bigk = bigk.to(torch.float).to(device)#(h,Thuge,d)
        bigv = bigv.to(torch.float).to(device)
        bigq = bigq.to(torch.float).to(device)


        list_projk = []
        list_projq = []
        list_projv = []
        list_projw = []
        list_sk = []
        list_sv = []
        #batching SVDs does not work because we are allocating too much mem
        for i in range(hkv):
            print("     ",i)
            Ki = bigk[i]
            Qi = bigq[i]
            

            Uk,Sk,Vk = torch.linalg.svd(Ki,full_matrices=False)
            Uq,Sq,Vq = torch.linalg.svd(Ki,full_matrices=False)

            del Qi,Ki,Uk,Uq,Sq
            torch.cuda.empty_cache()

            list_sk.append(Sk)
            projk = Vk[:,:r]
            projq = Vq[:,:r] @ Vq[:,:r].T @ Vk[:,:r]

            del Sk,Vk,Vq
            torch.cuda.empty_cache()

            

            


            Vi = bigv[i]

            Uv,Sv,Vv = torch.linalg.svd(Vi,full_matrices=False)

            del Vi,Uv
            torch.cuda.empty_cache()

            list_sv.append(Sv)
            projv = Vv[:,:r]
            projw = Vv[:,:r]

            del Sv,Vv
            torch.cuda.empty_cache()

            
            list_projk.append(projk)
            list_projq.append(projq)
            print("diff to id KQ:",torch.norm(torch.eye(d).to(projk.device)-projk@(projq.T)))
            list_projv.append(projv)
            list_projw.append(projw)
            print("diff to id VW:",torch.norm(torch.eye(d).to(projv.device)-projv@(projw.T)))
            
        
        del bigk,bigv,bigq
        torch.cuda.empty_cache()

        projk = torch.stack(list_projk,dim=0)
        projq = torch.stack(list_projq,dim=0)
        projv = torch.stack(list_projv,dim=0)
        projw = torch.stack(list_projw,dim=0)
        sk = torch.stack(list_sk,dim=0)
        sv = torch.stack(list_sv,dim=0)

        del list_projk,list_projv,list_projq,list_projw
        torch.cuda.empty_cache()

        if save:
            print("saving")
            pathbase = os.path.join(resname, str(l))
            os.makedirs(pathbase,exist_ok=True)
            torch.save(sk,os.path.join(pathbase,"speck.pt"))
            torch.save(sv,os.path.join(pathbase,"specv.pt"))

            torch.save(projk,os.path.join(pathbase,"projk.pt"))
            torch.save(projq,os.path.join(pathbase,"projq.pt"))
            torch.save(projv,os.path.join(pathbase,"projv.pt"))
            torch.save(projw,os.path.join(pathbase,"projw.pt"))

        del projk,projq,projv,projw,sk,sv
        torch.cuda.empty_cache()