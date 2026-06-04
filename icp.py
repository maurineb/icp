from pytorch3d.ops import knn_points
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np








#######################################################################################################################################################################################
################################################################################ FONCTIONS POUR LES DONNEES ###########################################################################
#######################################################################################################################################################################################
def read_file(filename, device):
    """
    Reads data from file .xyz and converts it to a tensor.

    Args:
        filename (string): Name of file to read.
        device (torch.device): Device to use.

    Returns:
        torch.Tensor: Point cloud. (N, 3)
    """
    pcl = np.loadtxt(filename)
    pcl = torch.FloatTensor(pcl)
    pcl = pcl.to(device)
    return pcl



def homogeneous_coord(pcl):
    """
    Adds a line of ones in order to transform the coordinates into homogeneous coordinates.

    Args:
        pcl (torch.Tensor): Point cloud. (3, N)

    Returns:
        torch.Tensor: Point cloud with homogeneous coordinates. (4, N)
    """
    assert(pcl.shape[0] == 3)
    return torch.vstack([pcl, torch.tensor([1] * pcl.shape[1], dtype=torch.float, device=pcl.device)])



def random_downsample(points, N):
    """
    Downsample a point cloud randomly.

    Args:
        points (torch.Tensor): Clean point cloud. (P, D)
        N (int): Number of points to downsample.

    Returns:
        torch.Tensor: Downsampled point cloud. (N, D)
    """
    P = points.shape[0]
    if N > P:
        return points
    else:
        indices = torch.randperm(P, device=points.device)[:N]
        return points[indices]



def add_gaussian_noise(points, std=torch.tensor(1.0), mean=torch.tensor(0.0)):
    """
    Add gaussian noise to a point cloud.

    Args:
        points (torch.Tensor): Clean point cloud. (D, N)
        std (torch.Tensor): Standard deviation of the noise. (0,)
        mean (torch.Tensor): Mean of the noise. (0,)

    Returns:
        torch.Tensor: Noisy point cloud. (D, N)
    """
    if std != torch.tensor(0.0):
        noise = torch.randn_like(points, device=points.device) * std + mean
        noisy_points = points + noise
        return noisy_points
    else:
        return points.clone()
    

    
def compute_rotation(thetaX, thetaY, thetaZ):
    """
    Computes the rotation matrix (along all 3D axis).

    Args:
        thetaX (torch.Tensor): Rotation angle in radians. (1,)
        thetaY (torch.Tensor): Rotation angle in radians. (1,)
        thetaZ (torch.Tensor): Rotation angle in radians. (1,)

    Returns:
        torch.Tensor: Rotation matrix. (3, 3)
    """
    R = torch.tensor([[ torch.cos(thetaY) * torch.cos(thetaZ)                                                            ,   -torch.cos(thetaY) * torch.sin(thetaZ)                                                            ,   torch.sin(thetaY)                     ],
                           [ torch.sin(thetaX) * torch.sin(thetaY) * torch.cos(thetaZ) + torch.cos(thetaX) * torch.sin(thetaZ),   -torch.sin(thetaX) * torch.sin(thetaY) * torch.sin(thetaZ) + torch.cos(thetaX) * torch.cos(thetaZ),  -torch.sin(thetaX) * torch.cos(thetaY) ],
                           [-torch.cos(thetaX) * torch.sin(thetaY) * torch.cos(thetaZ) + torch.sin(thetaX) * torch.sin(thetaZ),    torch.cos(thetaX) * torch.sin(thetaY) * torch.sin(thetaZ) + torch.sin(thetaX) * torch.cos(thetaZ),   torch.cos(thetaX) * torch.cos(thetaY)]],
                           device=thetaX.device)
    return R



def inject_outliers(y, outlier_ratio, bbox_scale=1.1):
    """
    Replace a fraction of points in y by outliers.

    Args:
        y (Tensor): (K, 3)
        outlier_ratio (float): in [0, 1]
        bbox_scale (float): enlarge bounding box around y

    Returns:
        y_out (Tensor): (K, 3) with outliers injected
        outlier_mask (BoolTensor): (K,) True where point is an outlier
    """
    K = y.shape[0]
    nb_out = int(round(outlier_ratio * K))
    if nb_out == 0:
        return y, torch.zeros(K, dtype=torch.bool, device=y.device)

    # indices to replace
    idx = torch.randperm(K, device=y.device)[:nb_out]

    # bounding box of current y
    y_min = y.min(dim=0).values
    y_max = y.max(dim=0).values
    center = 0.5 * (y_min + y_max)
    half_range = 0.5 * (y_max - y_min)

    # avoid degenerate ranges
    eps = 1e-8
    half_range = torch.clamp(half_range, min=eps)

    # enlarge box
    half_range = half_range * bbox_scale

    # uniform in enlarged bbox
    out = center + (torch.rand(nb_out, 3, device=y.device) * 2 - 1) * half_range

    y_out = y.clone()
    y_out[idx] = out

    mask = torch.zeros(K, dtype=torch.bool, device=y.device)
    mask[idx] = True
    return y_out, mask
    

    
def create_data(x_gt, sigma1, sigma2, N, M, outliers1=0.0, outliers2=0.0, max_trans=10, max_rot=3.14 / 12, rot=None, trans=None):
    """
    Generate data y1 and y2 from a clean point cloud x_gt.

    Args:
        x_gt (torch.Tensor): Clean point cloud. (P, 3)
        sigma1 (torch.Tensor): Standard deviation of the noise for y1. (0,)
        sigma2 (torch.Tensor): Standard deviation of the noise for y2. (0,)
        N (int): Number of points of y1.
        M (int): Number of points of y2.
        outliers1 (float): Outliers probability for y1. (0,)
        outliers2 (float): Outliers probability for y2. (0,)
        max_trans (torch.Tensor): Maximum translation between the data. (0,)
        max_rot (torch.Tensor): Maximum rotation angle between the data. (0,)

    Returns:
        torch.Tensor: Generated data y1. (N, 3)
        torch.Tensor: Generated data y2. (M, 3)
        torch.Tensor: Transformation matrix between y1 and y2. (4, 4)
    """
    # Downsampling
    y1 = random_downsample(x_gt, N)
    y2 = random_downsample(x_gt, M)

    # Add gaussian noise
    _, _, scale_1 = normalize_unit_sphere(y1)
    _, _, scale_2 = normalize_unit_sphere(y2)
    y1 = add_gaussian_noise(y1, sigma1*scale_1)
    y2 = add_gaussian_noise(y2, sigma2*scale_2)

    # Compute random transformation
    T_gt = torch.eye(4, dtype=torch.float, device=x_gt.device)
    if rot == None:
        angles = (torch.rand(3, device=x_gt.device) * 2 - 1) * max_rot
        thetaX, thetaY, thetaZ = angles
        rot = compute_rotation(thetaX, thetaY, thetaZ)
    if trans == None:
        trans = (torch.rand(3, device=x_gt.device) - 0.5) * 2 * max_trans

    # Apply transformation on y1
    y1 = y1 @ rot.T + trans
    T_gt[:3, :3] = rot
    T_gt[:3, 3] = trans

    # Adding outliers
    y1, _ = inject_outliers(y1, outliers1)
    y2, _ = inject_outliers(y2, outliers2)

    return y1, y2, T_gt



def normalize_unit_sphere(pcl, center=None, scale=None):
    """
    Args:
        pcl:  The point cloud to be normalized, (N, 3)
    """
    if center is None:
        p_max = pcl.max(dim=0, keepdim=True)[0]
        p_min = pcl.min(dim=0, keepdim=True)[0]
        center = (p_max + p_min) / 2    # (1, 3)
    pcl = pcl - center
    if scale is None:
        scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
    pcl = pcl / scale
    return pcl, center, scale



def adapt_transformation(T, sigma_1, mean_1, sigma_2, mean_2):
    """
    Adqpt the transformation matrix to the normalized data.

    Args:
        T (torch.Tensor): Transformation matrix between point clouds y1 and y2 before normalization. (4, 4)
        sigma_1 (Float): Scale for the normalization of y1.
        mean_1 (torch.Tensor): Center for the normalization of y1. (3,)
        sigma_2 (Float): Scale for the normalization of y2.
        mean_2 (torch.Tensor): Center for the normalization of y2. (3,)

    Returns:
        torch.Tensor: New transformation matrix. (4, 4)
    """
    new_T = torch.eye(4, 4, device=T.device)
    new_T[:3, :3] = T[:3, :3] * sigma_2/sigma_1
    new_T[:3, 3] = (T[:3, 3] + torch.matmul(T[:3, :3], mean_2) - mean_1)/sigma_1
    return new_T
#######################################################################################################################################################################################
#######################################################################################################################################################################################
#######################################################################################################################################################################################




def nearest_neighbors(target, source_transformed):
    """
    Nearest neighbors of source_transformed.

    Inputs :
        target: (3, N) torch.Tensor
        source_transformed: (3, M) torch.Tensor

    Output :
        idx: (N) torch.Tensor
    """
    nn = knn_points(target.T.unsqueeze(0), source_transformed.T.unsqueeze(0), K=1, return_nn=False)
    idx = nn.idx.squeeze(0).squeeze(-1)        # (N)
    return idx



def rotation_error(R1, R2):
    """
    Computes the rotation error between R1 and R2.

    Args:
        R1 (torch.Tensor): First rotation matrix. (3, 3)
        R2 (torch.Tensor): Second rotation matrix. (3, 3)

    Returns:
        torch.Tensor: Rotation error between the two rotation matrices. (0,)
    """
    R_err = R1.T @ R2
    cos_theta = (torch.trace(R_err) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    return torch.rad2deg(theta)



def translation_error(t1, t2):
    """
    Computes the translation error between t1 and t2.

    Args:
        t1 (torch.Tensor): First translation vector. (3)
        t2 (torch.Tensor): Second translation vector. (3)

    Returns:
        torch.Tensor: Translation error between the two translation vectors. (0,)
    """
    return torch.norm(t1 - t2)



def estimate_T_svd(x, y, min_H=1e-9):
    """
    Estimate a rigid transform T in SE(3) with standard SVD / Procrustes:
        min_{R,t} sum_i || y_i - (R x_i + t) ||^2

    Args:
        x: (N, 3) source points
        y: (N, 3) target points

    Returns:
        T: (4, 4) rigid transform such that y ~= R x + t
    """
    assert x.dim() == 2 and x.shape[1] == 3
    assert y.dim() == 2 and y.shape[1] == 3
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] != 0

    device = x.device
    dtype = x.dtype

    # Centroids
    x_bar = x.mean(dim=0)   # (3,)
    y_bar = y.mean(dim=0)   # (3,)

    # Centered points
    x_c = x - x_bar         # (N,3)
    y_c = y - y_bar         # (N,3)

    # Cross-covariance
    H = x_c.transpose(0, 1) @ y_c   # (3,3)

    # Degenerate case
    H_norm = torch.linalg.norm(H)
    if H_norm.item() < min_H:
        T = torch.eye(4, device=device, dtype=dtype)
        T[:3, 3] = y_bar - x_bar
        return T

    U, S, Vh = torch.linalg.svd(H)
    V = Vh.T

    R = (V @ U.T).to(dtype=dtype)

    # Reflection fix
    if torch.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = (V @ U.T).to(dtype=dtype)

    # Translation
    t = y_bar - R @ x_bar

    # Build T
    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T



def icp(source, target, T_gt, max_iterations=20, association_function=nearest_neighbors):
    """
    ICP in PyTorch to align source on target.

    Inputs :
        source: (4, N) torch.Tensor
        target: (4, M) torch.Tensor
        max_iterations: int

    Output :
        transform: transformation matrix 4x4
        source_transformed: source with the transformation applied (4, N) torch.Tensor
    """
    device = source.device
    _, N = source.shape
    _, M = target.shape

    # Initialization of the transformation matrix
    transform = torch.eye(4, device=device)  # (4, 4)
    source_transformed = source.clone()

    RE = []
    TE = []

    for i in tqdm(range(max_iterations)):
        # FIST STEP: ASSOCIATION
        #######################################################################################################################
        idx = association_function(source_transformed[:3,:], target[:3,:])
        target_mapped = target[:, idx]
        #######################################################################################################################

        # SECOND STEP: ESTIMATION OF THE TRANSFORMATION
        #######################################################################################################################
        T_iter = estimate_T_svd(source_transformed[:3,:].T, target_mapped[:3,:].T)
        source_transformed = T_iter @ source_transformed
        transform = T_iter @ transform
        #######################################################################################################################

        # Compute errors
        RE_i = rotation_error(T_gt[:3,:3], transform[:3,:3])
        TE_i = translation_error(T_gt[:3,3], transform[:3,3])
        RE.append(RE_i.item())
        TE.append(TE_i.item())

    return transform, source_transformed, RE, TE










if __name__ == "__main__":
    # Loading the data
    device = torch.device("cpu")
    filename = "/projects/UDIP/maurine/recalage_tweedie/3D-registration-tweedie/data/lidar_velodrome_marseille.xyz"
    x_gt = read_file(os.path.join(filename), device)

    # Generate the observed data
    sigma1 = torch.tensor(0.01)
    sigma2 = torch.tensor(0.01)
    prop_out1 = 0.0
    prop_out2 = 0.0
    x_gt = random_downsample(x_gt, 10000)
    y1, y2, T_gt = create_data(x_gt, sigma1, sigma2, 10000, 10000, outliers1=prop_out1, outliers2=prop_out2)


    # Normalizing
    y2, mean_points, sigma_points = normalize_unit_sphere(y2)
    x_gt = (x_gt - mean_points) / sigma_points
    mean_points = mean_points.to(device)
    sigma_points = sigma_points.to(device)
    y1, mean_points_1, sigma_points_1 = normalize_unit_sphere(y1)
    y1 = y1 * sigma_points_1 / sigma_points

    # Adding a line of ones
    x_gt = homogeneous_coord(x_gt.T)
    y1 = homogeneous_coord(y1.T)
    y2 = homogeneous_coord(y2.T)

    # Adapt the transformation matrix to the normalized data with two different means for y1 and y2
    T_gt = adapt_transformation(T_gt, sigma_points.item(), mean_points_1.squeeze(), sigma_points.item(), mean_points.squeeze())








    # ICP
    # Parameters
    iter = 100
    T, y2_reg, RE, TE = icp(y2, y1, T_gt, max_iterations=iter, association_function=nearest_neighbors)




    # Affichage des nuages de points
    plt.figure(f"Initial point clouds")
    axes = plt.axes(projection="3d")
    y1_np = y1.cpu().numpy()
    y2_np = y2.cpu().numpy()
    axes.scatter(y1_np[0, :], y1_np[1, :], y1_np[2, :], c="red", s=10, alpha=0.2, marker='.', label="First point cloud")
    axes.scatter(y2_np[0, :], y2_np[1, :], y2_np[2, :], c="dodgerblue", s=10, alpha=0.2, marker='.', label="Second point cloud")
    axes.set_title(f"Initial point clouds")
    plt.show()
    plt.close()


    plt.figure(f"Registered point clouds")
    axes = plt.axes(projection="3d")
    y2_reg_np = y2_reg.cpu().numpy()
    axes.scatter(y1_np[0, :], y1_np[1, :], y1_np[2, :], c="red", s=10, alpha=0.2, marker='.', label="Initial point cloud")
    axes.scatter(y2_reg_np[0, :], y2_reg_np[1, :], y2_reg_np[2, :], c="dodgerblue", s=10, alpha=0.2, marker='.', label="Estimated point cloud")
    axes.set_title(f"Registered point cloud")
    plt.show()
    plt.close()


    plt.figure(f"Registered point clouds")
    axes = plt.axes(projection="3d")
    Ty1_np = torch.matmul(torch.inverse(T), y1).detach().cpu().numpy()
    axes.scatter(y2_np[0, :], y2_np[1, :], y2_np[2, :], c="dodgerblue", s=10, alpha=0.2, marker='.', label="Initial point cloud")
    axes.scatter(Ty1_np[0, :], Ty1_np[1, :], Ty1_np[2, :], c="red", s=10, alpha=0.2, marker='.', label="Estimated point cloud")
    axes.set_title(f"Registered point clouds")
    plt.show()
    plt.close()





    # Affichage des courbes des erreurs au cours des itérations
    start_loss = 0
    end_loss = iter
    iter_list = list(range(start_loss, end_loss))

    # Display the difference R and the ground truth
    fig, axes = plt.subplots()
    axes.plot(iter_list, RE[start_loss:end_loss], label="icp")
    axes.set_title("Evolution of the RE between R and the groundtruth")
    axes.set_xlabel("iteration")
    axes.set_ylabel("RE")
    plt.show()
    plt.close()


    # Display the difference t and the ground truth
    fig, axes = plt.subplots()
    axes.plot(iter_list, TE[start_loss:end_loss], label="icp")
    axes.set_title("Evolution of the TE between t and the groundtruth")
    axes.set_xlabel("iteration")
    axes.set_ylabel("TE")
    plt.show()
    plt.close()





