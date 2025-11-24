import torch
import numpy as np
import cv2
import os
import numpy as np
import cv2
import torch


class MaskExtractor:
    def __init__(self, num_masks=3):
        """
        Initializes the MaskExtractor with the number of masks to extract.

        :param num_masks: Number of masks to extract.
        """
        self.num_masks = num_masks

    def extract_masks(self, image):
        """
        Extracts up to `self.num_masks` connected component masks from the input image,
        sorted by the centroids of the connected components.

        :param image: Input grayscale image (output from Model1).
        :return: List of up to `self.num_masks` masks based on connected components.
        """
        # Convert PyTorch tensor to NumPy array if necessary
        # if isinstance(image, torch.Tensor):
        #     image = image.cpu().numpy()

        # Ensure image values are within [0, 1] and convert to uint8
        if image.dtype != np.uint8:
            image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)

        # Binarize image for connected component labeling
        _, binary_image = cv2.threshold(image[0, 0, :, :], 127, 255, cv2.THRESH_BINARY)

        # Label connected components using OpenCV
        num_labels, labels = cv2.connectedComponents(binary_image)

        # List to store masks and their centroids
        masks_with_centroids = []

        for label in range(1, num_labels):
            if len(masks_with_centroids) >= self.num_masks:
                break

            # Get the mask for the current connected component
            mask = (labels == label).astype(np.uint8)

            # Extract original pixel values for the mask
            mask_image = np.where(mask == 1, image, 0).astype(np.uint8)
            # mask[labels == label] = image[labels == label]

            # Compute the centroid of the mask
            M = cv2.moments(mask)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])

                masks_with_centroids.append((mask_image, (centroid_x, centroid_y)))

        # If fewer masks are found than requested, pad the list with zeros
        while len(masks_with_centroids) < self.num_masks:
            masks_with_centroids.append((np.zeros_like(image, dtype=np.uint8), (float('inf'), float('inf'))))

        # Sort masks based on the centroids
        masks_with_centroids.sort(key=lambda x: (x[1][0], x[1][1]))

        # Extract sorted masks
        sorted_masks = [torch.tensor(mask).unsqueeze(0) for mask, _ in masks_with_centroids]


        return sorted_masks


