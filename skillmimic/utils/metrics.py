import torch

class Metrics:
    def __init__(self) -> None:
        pass
 
    def compute_evaluation_metrics(hoi_ref, hoi_obs, contact_buf, tar_contact_forces, len_keypos): #metric zqh
        ### data preprocess ###
        # simulated states

        root_pos = hoi_obs[:,:3]
        root_rot = hoi_obs[:,3:3+4]
        dof_pos = hoi_obs[:,7:7+51*3]
        dof_pos_vel = hoi_obs[:,160:160+51*3]

        obj_pos = hoi_obs[:,313:313+3]
        obj_rot = hoi_obs[:,316:316+4]
        obj_pos_vel = hoi_obs[:,320:320+3]
        key_pos = hoi_obs[:,323:323+len_keypos*3]
        key_pos = torch.cat((root_pos, key_pos),dim=-1)
        body_rot = torch.cat((root_rot, dof_pos),dim=-1)
        ig = key_pos.view(-1,len_keypos+1,3).transpose(0,1) - obj_pos[:,:3]
        ig = ig.transpose(0,1).view(-1,(len_keypos+1)*3)
        # object contact
        contact = hoi_obs[:,-1:]# fake one
        obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1).to(torch.int) # =1 when contact happens to the object
        # body contact
        contact_body_ids = [0,1,2,5,6,9,10,11,12,13,14,15,16,17,34,35,36]
        body_contact_buf = contact_buf[:, contact_body_ids, :].clone() #env, 
        body_contact = ((torch.abs(body_contact_buf) < 0.1).prod(dim=-1).prod(dim=-1) == 1).to(torch.int) # =1 when no contact happens to the body

        # reference states
        ref_root_pos = hoi_ref[:,:3]
        ref_root_rot = hoi_ref[:,3:3+4]
        ref_dof_pos = hoi_ref[:,7:7+51*3]
        ref_dof_pos_vel = hoi_ref[:,160:160+51*3]
        ref_obj_pos = hoi_ref[:,313:313+3]
        ref_obj_rot = hoi_ref[:,316:316+4]
        ref_obj_pos_vel = hoi_ref[:,320:320+3]
        ref_key_pos = hoi_ref[:,323:323+len_keypos*3]
        ref_key_pos = torch.cat((ref_root_pos, ref_key_pos),dim=-1)
        ref_body_rot = torch.cat((ref_root_rot, ref_dof_pos),dim=-1)
        ref_ig = ref_ig.transpose(0,1).view(-1,(len_keypos+1)*3)
        ref_ig = ref_key_pos.view(-1,len_keypos+1,3).transpose(0,1) - ref_obj_pos[:,:3]
        # object contact
        ref_obj_contact = hoi_ref[:,-1:].to(torch.int)
        # body contact
        ref_body_contact = torch.ones_like(ref_obj_contact, dtype=torch.int) # no body contact for all time 

        # Position Error

        # The Mean Per-Joint Position Error (MPJPE) of the humanoid ($E_\text{b-mpjpe}$) and object ($E_\text{o-mpjpe}$) is used to evaluate the positional imitation performance (in $mm$) following
        ## Body
        pos_error_body = (ref_key_pos.reshape(-1, len_keypos+1, 3) - key_pos.reshape(-1, len_keypos+1, 3)).norm(dim=-1).mean(dim=-1)
        # pos_error_body = torch.mean(torch.abs(ref_key_pos - key_pos), dim=-1)
        ## Ball
        pos_error_ball = (ref_obj_pos - obj_pos).norm(dim=-1)
        # pos_error_ball = torch.mean(torch.abs(ref_obj_pos - obj_pos), dim=-1)

        # Accuracy
        # The overall accuracy of HOI imitation, abbreviated as \textit{Acc}., is defined per frame, and deems imitation accurate when the object position and body position errors are both under the thresholds and the contacts are correct. 

        # The object threshold is defined as 0.2$m$. The body threshold is defined as 0.1$m$. The \textit{Acc} is calculated by averaging the success values of all frames.
        ## Body
        pos_acc_body = torch.where(pos_error_body < 0.1, torch.ones_like(pos_error_body, dtype=torch.bool), torch.zeros_like(pos_error_body, dtype=torch.bool))
        ## Ball
        pos_acc_ball = torch.where(pos_error_ball < 0.2, torch.ones_like(pos_error_ball, dtype=torch.bool), torch.zeros_like(pos_error_ball, dtype=torch.bool))

        ## Contact
        contact_correctness = (~(torch.logical_xor(body_contact, ref_body_contact[:,0]))) & (~(torch.logical_xor(obj_contact, ref_obj_contact[:,0])))

        accuracy = pos_acc_body & pos_acc_ball & contact_correctness
        # print(pos_acc_body, pos_acc_ball, contact_correctness)


        # Contact Error
        # The contact error $E_\text{cg}$, ranging from 0 to 1, is defined as $\frac{1}{N}\sum_{t=1}^{N}\text{MSE}(\boldsymbol{s}_{t}^{cg},\hat{\boldsymbol{s}}_{t}^{cg})$, where N is the total frames of the reference HOI data. 

        contact_sim = torch.stack((body_contact, obj_contact), dim=1).to(torch.float)
        contact_gt = torch.cat((ref_body_contact, ref_obj_contact), dim=1).to(torch.float)
        contact_error = ((contact_sim-contact_gt)**2).mean(dim=-1)


        return accuracy.to(torch.float), pos_error_body.to(torch.float)*1000., pos_error_ball.to(torch.float)*1000., contact_error.to(torch.float)


# @torch.jit.script
def compute_evaluation_metrics(hoi_ref, hoi_obs, contact_buf, tar_contact_forces, len_keypos): #metric zqh
    # type: (Tensor, Tensor, Tensor, Tensor, int) ->  Tuple[Tensor, Tensor, Tensor, Tensor]

    ### data preprocess ###

    # simulated states
    root_pos = hoi_obs[:,:3]
    root_rot = hoi_obs[:,3:3+4]
    dof_pos = hoi_obs[:,7:7+51*3]
    dof_pos_vel = hoi_obs[:,160:160+51*3]
    obj_pos = hoi_obs[:,313:313+3]
    obj_rot = hoi_obs[:,316:316+4]
    obj_pos_vel = hoi_obs[:,320:320+3]
    key_pos = hoi_obs[:,323:323+len_keypos*3]
    key_pos = torch.cat((root_pos, key_pos),dim=-1)
    body_rot = torch.cat((root_rot, dof_pos),dim=-1)
    ig = key_pos.view(-1,len_keypos+1,3).transpose(0,1) - obj_pos[:,:3]
    ig = ig.transpose(0,1).view(-1,(len_keypos+1)*3)
    # object contact
    contact = hoi_obs[:,-1:]# fake one
    obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1).to(torch.int) # =1 when contact happens to the object
    # body contact
    contact_body_ids = [0,1,2,5,6,9,10,11,12,13,14,15,16,17,34,35,36]
    body_contact_buf = contact_buf[:, contact_body_ids, :].clone() #env, 
    body_contact = ((torch.abs(body_contact_buf) < 0.1).prod(dim=-1).prod(dim=-1) == 1).to(torch.int) # =1 when no contact happens to the body
    
    # reference states
    ref_root_pos = hoi_ref[:,:3]
    ref_root_rot = hoi_ref[:,3:3+4]
    ref_dof_pos = hoi_ref[:,7:7+51*3]
    ref_dof_pos_vel = hoi_ref[:,160:160+51*3]
    ref_obj_pos = hoi_ref[:,313:313+3]
    ref_obj_rot = hoi_ref[:,316:316+4]
    ref_obj_pos_vel = hoi_ref[:,320:320+3]
    ref_key_pos = hoi_ref[:,323:323+len_keypos*3]
    ref_key_pos = torch.cat((ref_root_pos, ref_key_pos),dim=-1)
    ref_body_rot = torch.cat((ref_root_rot, ref_dof_pos),dim=-1)
    ref_ig = ref_key_pos.view(-1,len_keypos+1,3).transpose(0,1) - ref_obj_pos[:,:3]
    ref_ig = ref_ig.transpose(0,1).view(-1,(len_keypos+1)*3)
    # object contact
    ref_obj_contact = hoi_ref[:,-1:].to(torch.int)
    # body contact
    ref_body_contact = torch.ones_like(ref_obj_contact, dtype=torch.int) # no body contact for all time 

    
    # Position Error
    # The Mean Per-Joint Position Error (MPJPE) of the humanoid ($E_\text{b-mpjpe}$) and object ($E_\text{o-mpjpe}$) is used to evaluate the positional imitation performance (in $mm$) following
    ## Body
    pos_error_body = (ref_key_pos.reshape(-1, len_keypos+1, 3) - key_pos.reshape(-1, len_keypos+1, 3)).norm(dim=-1).mean(dim=-1)
    # pos_error_body = torch.mean(torch.abs(ref_key_pos - key_pos), dim=-1)
    ## Ball
    pos_error_ball = (ref_obj_pos - obj_pos).norm(dim=-1)
    # pos_error_ball = torch.mean(torch.abs(ref_obj_pos - obj_pos), dim=-1)


    # Accuracy
    # The overall accuracy of HOI imitation, abbreviated as \textit{Acc}., is defined per frame, and deems imitation accurate when the object position and body position errors are both under the thresholds and the contacts are correct. 
    # The object threshold is defined as 0.2$m$. The body threshold is defined as 0.1$m$. The \textit{Acc} is calculated by averaging the success values of all frames.
    ## Body
    pos_acc_body = torch.where(pos_error_body < 0.1, torch.ones_like(pos_error_body, dtype=torch.bool), torch.zeros_like(pos_error_body, dtype=torch.bool))

    ## Ball
    pos_acc_ball = torch.where(pos_error_ball < 0.2, torch.ones_like(pos_error_ball, dtype=torch.bool), torch.zeros_like(pos_error_ball, dtype=torch.bool))

    ## Contact
    contact_correctness = (~(torch.logical_xor(body_contact, ref_body_contact[:,0]))) & (~(torch.logical_xor(obj_contact, ref_obj_contact[:,0])))

    accuracy = pos_acc_body & pos_acc_ball & contact_correctness
    # print(pos_acc_body, pos_acc_ball, contact_correctness)


    # Contact Error
    # The contact error $E_\text{cg}$, ranging from 0 to 1, is defined as $\frac{1}{N}\sum_{t=1}^{N}\text{MSE}(\boldsymbol{s}_{t}^{cg},\hat{\boldsymbol{s}}_{t}^{cg})$, where N is the total frames of the reference HOI data. 
    contact_sim = torch.stack((body_contact, obj_contact), dim=1).to(torch.float)
    contact_gt = torch.cat((ref_body_contact, ref_obj_contact), dim=1).to(torch.float)
    contact_error = ((contact_sim-contact_gt)**2).mean(dim=-1)


    return accuracy.to(torch.float), pos_error_body.to(torch.float)*1000., pos_error_ball.to(torch.float)*1000., contact_error.to(torch.float)
