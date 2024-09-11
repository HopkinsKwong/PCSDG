# The HD95 calculation code in the following section is borrowed from the metric.py file of the medpy library.
# medpy project repository: https://github.com/loli/medpy

from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
import numpy as np
import torch
from sklearn.neighbors import KDTree
from scipy import ndimage
import GeodisTK

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    ## change into binary type
    zero = torch.zeros_like(result)
    one = torch.ones_like(result)
    result = torch.where(result > 0.5, one, zero)
    reference = torch.where(reference > 0.5, one, zero)
    
    result = result.numpy()
    reference = reference.numpy()
    result = result.astype(int)
    reference = reference.astype(int)

    # result = numpy.atleast_1d(result.astype(bool))
    # reference = numpy.atleast_1d(reference.astype(bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = numpy.asarray(voxelspacing, dtype=numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == numpy.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == numpy.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = numpy.percentile(numpy.hstack((hd1, hd2)), 95)
    return hd95

def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim == 2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge

def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    ## change into binary type
    zero = torch.zeros_like(s)
    one = torch.ones_like(s)
    s = torch.where(s > 0.5, one, zero)
    g = torch.where(g > 0.5, one, zero)
    s = s.numpy()
    g = g.numpy()
    s = s.astype(np.uint8)
    g = g.astype(np.uint8)

    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    if int(len(dist_list1) * 0.95) < 1:
        dist1 = 100
    else:
        dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    if int(len(dist_list2) * 0.95) < 1:
        dist2 = 100
    else:
        dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)

def get_hard_hd95(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)

    hd95_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        ## When the sum of pixels of a certain category in the prediction result is too small, the denominator may be zero or close to zero,
        # causing a divide by zero error or numerical instability.
        hd95_list.append(binary_hausdorff95(output, mask))
    if return_list:
        return np.mean(hd95_list), hd95_list
    else:
        return np.mean(hd95_list)

def get_dice_threshold(output, mask, threshold=0.5):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: dice of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    intersection = (output * mask).sum()
    dice = (2. * intersection + smooth) / (output.sum() + mask.sum() + smooth)

    return dice


def get_hard_dice(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    dice_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        dice_list.append(get_dice_threshold(output, mask, threshold=0.5))
    if return_list:
        return np.mean(dice_list), dice_list
    else:
        return np.mean(dice_list)

# IOU实现
def get_iou_threshold(output, mask, threshold=0.5):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: iou of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    intersection = (mask * output).sum()
    union = (mask + output).sum() - intersection
    iou = intersection / union if union > 0 else 0.0

    return iou

def get_hard_iou(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    iou_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        iou_list.append(get_iou_threshold(output, mask, threshold=0.5))
    if return_list:
        return np.mean(iou_list), iou_list
    else:
        return np.mean(iou_list)

# ASSD,MSSD
def get_assd_threshold(output, mask, threshold=0.5):
    struct = ndimage.generate_binary_structure(2, 1)

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)

    output = output.to(torch.int32)
    mask = mask.to(torch.int32)

    ref_border = mask ^ ndimage.binary_erosion(mask, struct, border_value=1)
    ref_border_voxels = np.array(np.where(ref_border)).T  # 获取gt边界点的坐标,为一个n*dim的数组

    seg_border = output ^ ndimage.binary_erosion(output, struct, border_value=1)
    seg_border_voxels = np.array(np.where(seg_border)).T  # 获取seg边界点的坐标,为一个n*dim的数组

    # print(f"ref_border_voxels.shape: {ref_border_voxels.shape}")
    # print(f"seg_border_voxels.shape: {seg_border_voxels.shape}")
    tree_ref = KDTree(np.array(ref_border_voxels))
    dist_seg_to_ref, ind = tree_ref.query(seg_border_voxels)
    tree_seg = KDTree(np.array(seg_border_voxels))
    dist_ref_to_seg, ind2 = tree_seg.query(ref_border_voxels)

    assd = (dist_seg_to_ref.sum() + dist_ref_to_seg.sum()) / (len(dist_seg_to_ref) + len(dist_ref_to_seg))
        # mssd=np.concatenate((dist_seg_to_ref, dist_ref_to_seg)).max()
    return assd


def get_assd(outputs, masks, return_list=False):
    outputs = outputs.detach().to(torch.float64)
    masks = masks.detach().to(torch.float64)
    assd_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        assd_output = get_assd_threshold(output, mask, threshold=0.5)
        assd_list.append(get_assd_threshold(output, mask, threshold=0.5))
    if return_list:
        return np.mean(assd_list), assd_list
    else:
        return np.mean(assd_list)
