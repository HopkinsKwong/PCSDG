def normalize_image(img_npy):
    """
    :param img_npy: b, c, h, w
    """
    for b in range(img_npy.shape[0]):
        for c in range(img_npy.shape[1]):
            img_npy[b, c] = (img_npy[b, c] - img_npy[b, c].mean()) / (img_npy[b, c].std() + 1e-5)
    return img_npy


def normalize_image_to_0_1(img):
    return (img-img.min())/(img.max()-img.min())


def normalize_image_to_m1_1(img):
    return -1 + 2 * (img-img.min())/(img.max()-img.min())


def ct_normalize_image(img):
    ct_mean, ct_std = 47.34859836890016,62.98354855180914
    """
    Normalizing CT images, based on global statistics in advance
    """
    return (img - ct_mean) / ct_std

def mr_normalize_image(img):
    return (img - img.mean()) / img.std()