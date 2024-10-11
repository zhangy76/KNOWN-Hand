def compute_anglebound(pred_pose, angle_limits, batch_size, interjoint, device):
    # update joint range of motion given different joint angles using the function anatomy knowledge

    angle_limits_bound = angle_limits.expand(2, batch_size, 48).clone()

    if interjoint == 0:
        return angle_limits_bound
    # eliminate prediction that are outside the original bound
    pred_pose_bound = pred_pose.clone().detach()
    pred_pose_bound = torch.where(pred_pose_bound>angle_limits_bound[0],pred_pose_bound,angle_limits_bound[0])
    pred_pose_bound = torch.where(pred_pose_bound<angle_limits_bound[1],pred_pose_bound,angle_limits_bound[1])

    # DIP on PIP
    DIP_index = [11,20,38,29] # [4,7,13,10]
    PIP_index = [8,17,35,26] # [3,6,12,9]
    angle_limits_bound[0,:,PIP_index] = torch.where(angle_limits_bound[0,:,DIP_index]>0, torch.zeros([batch_size,4]).to(device), angle_limits[0,:,PIP_index])

    # MCP joint fingers
    MCP_index_beta = [4,13,31,22] # [2,5,11,8]
    MCP_index_gamma = [5,14,32,23]
    angle_limits_bound[0,:,MCP_index_beta] = angle_limits_bound[0,:,MCP_index_beta] * (1 - (pred_pose_bound[:,MCP_index_gamma].clamp(max=70/180*np.pi)/(70/180*np.pi)))
    angle_limits_bound[1,:,MCP_index_beta] = angle_limits_bound[1,:,MCP_index_beta] * (1 - (pred_pose_bound[:,MCP_index_gamma].clamp(max=70/180*np.pi)/(70/180*np.pi)))
    # MCP joint thumb
    MCP_thumb_beta = 40 # 14
    MCP_thumb_gamma = 41
    angle_limits_bound[0,:,MCP_thumb_gamma] = angle_limits_bound[0,:,MCP_thumb_gamma] * (1 + (pred_pose_bound[:,MCP_thumb_beta].clamp(min=-70/180*np.pi)/(70/180*np.pi)))
    angle_limits_bound[1,:,MCP_thumb_gamma] = angle_limits_bound[1,:,MCP_thumb_gamma] * (1 + (pred_pose_bound[:,MCP_thumb_beta].clamp(min=-70/180*np.pi)/(70/180*np.pi)))

    # eliminate new bound that are outside the original bound
    angle_limits_bound[0] = torch.where(angle_limits_bound[0]>angle_limits[0],angle_limits_bound[0],angle_limits[0])
    angle_limits_bound[1] = torch.where(angle_limits_bound[1]<angle_limits[1],angle_limits_bound[1],angle_limits[1])

    return angle_limits_bound

def losses(pred_euler, interjoint, device):
    # pred_euler: Nx15x3 (no root joint rotation)
    # interjoint: scalar (if 0, not consider the functional anatomy knowledge)
    
    batch_size = pred_euler.shape[0]

    # biomechanics
    if handtype == 'RIGHT':
        angle_limits = torch.from_numpy(constants.BOF_RIGHT).to(device).float().unsqueeze(1)
    else:
        angle_limits = torch.from_numpy(constants.BOF_LEFT).to(device).float().unsqueeze(1)

    # functional anatomy
    angle_limits = compute_anglebound(pred_euler, angle_limits, batch_size, interjoint, device)

    anglegreater = pred_euler - angle_limits[1,:]
    anglesmaller = angle_limits[0,:] - pred_euler
    jointangle_loss = torch.sum(anglegreater.clamp(min=0)**2+anglesmaller.clamp(min=0)**2, dim=1) # N,

    return jointangle_loss




