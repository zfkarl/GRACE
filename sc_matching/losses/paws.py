import torch

def paws_loss(
    tau=0.1,
    T=0.25,
    me_max=True
):
    """
    Make semi-supervised PAWS loss

    :param multicrop: number of small multi-crop views
    :param tau: cosine similarity temperature
    :param T: target sharpenning temperature
    :param me_max: whether to perform me-max regularization
    """
    softmax = torch.nn.Softmax(dim=1)

    def sharpen(p):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def snn(query, supports, labels):
        """ Soft Nearest Neighbours similarity classifier """
        # Step 1: normalize embeddings
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

        # Step 2: compute similarlity between local embeddings
        return softmax(query @ supports.T / tau) @ labels

    def loss(
        anchor_views,           # unlabled rna
        anchor_supports,        # labled rna
        anchor_support_labels,  # rna labels
        target_views,           # unlabled atac
        target_supports,        # labled atac
        target_support_labels,  # atac labels
        sharpen=sharpen,
        snn=snn
    ):
        
        # Step 1: compute anchor predictions
        probs = snn(anchor_views, anchor_supports, anchor_support_labels)

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = snn(target_views, target_supports, target_support_labels)
            targets = sharpen(targets)
            targets[targets < 1e-4] *= 0  # numerical stability

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if me_max:
            avg_probs = torch.mean(sharpen(probs), dim=0)
            rloss -= torch.sum(torch.log(avg_probs**(-avg_probs)))
        loss = loss +rloss
        return loss
    
    def mutal_loss(
        anchor_views,          
        anchor_supports,       
        anchor_support_labels,  
        target_views,           
        target_supports,        
        target_support_labels,
        loss = loss
    ):
        loss_a = loss(anchor_views,anchor_supports, anchor_support_labels,  target_views,target_supports, target_support_labels )
        loss_b = loss(target_views,target_supports, target_support_labels, anchor_views,anchor_supports, anchor_support_labels)
        total_loss = (loss_a + loss_b)/2.
        return total_loss 
            
    return mutal_loss


