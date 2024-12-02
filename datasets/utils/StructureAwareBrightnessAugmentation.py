import numpy as np
import random
from scipy.special import comb
from PIL import Image

class Structure_Aware_Brightness_Augmentation(object):
    def __init__(self,pixel_range=(0.,255.)):
        self.pixel_range = pixel_range
        self.nPoints = 4
        self.nTimes = 100000
        self.slide_limit = 20
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array([bernstein_poly(i, self.nPoints - 1, t) for i in range(0, self.nPoints)]).astype(np.float32)

    def get_bezier_curve(self,points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def alpha_non_linear_transformation(self, image_brightness_inputs):
        target_min = 0.5 # 目标是把alpha映射到0.5~1.5
        start_point, end_point = image_brightness_inputs.min(), image_brightness_inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints - 2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        xvals, yvals = np.sort(xvals), np.sort(yvals)
        return (np.interp(image_brightness_inputs, xvals, yvals) / 255) + target_min # 0~255线性映射到0.5~1.5，这个返回值就是alpha_brightness

    def brightness_transformation(self,inputs,alpha_brightness):
        beta_contrast = np.array(random.gauss(0, 0.1) * 255, dtype=np.float32)
        # beta_contrast = np.clip(beta_contrast, self.pixel_range[0] - np.percentile(inputs, self.slide_limit),
        #                    self.pixel_range[1] - np.percentile(inputs, 100 - self.slide_limit))
        return np.clip(np.abs(inputs * alpha_brightness + beta_contrast), self.pixel_range[0], self.pixel_range[1])

    def Local_Brightness_Augmentation(self, img_npy, mask):
        batch_size, channels, height, width = img_npy.shape
        image_brightness = np.zeros((batch_size, 1, height, width), dtype=img_npy.dtype)
        image_brightness[:, 0, :, :] = 0.299 * img_npy[:, 0, :, :] + 0.587 * img_npy[:, 1, :, :] + 0.114 * img_npy[:, 2, :, :]
        # img_transposed = img_npy.transpose(1, 2, 0)
        # img_restored = img_transposed.astype(np.uint8)
        # image_brightness= Image.fromarray(img_restored).convert("L")
        # image_brightness = np.array(image_brightness)
        # image_brightness = image_brightness[np.newaxis]

        output_image = np.zeros_like(img_npy)
        mask = mask.astype(np.int32)
        mask = np.tile(mask, np.array(img_npy.shape) // np.array(mask.shape))
        image_brightness = np.tile(image_brightness, np.array(img_npy.shape) // np.array(image_brightness.shape))

        for c in range(0,np.max(mask)+1):
            if (mask==c).sum()==0:continue
            output_image[mask == c] = self.brightness_transformation(img_npy[mask == c],
                                                                     self.alpha_non_linear_transformation(image_brightness[mask == c]))

        return output_image