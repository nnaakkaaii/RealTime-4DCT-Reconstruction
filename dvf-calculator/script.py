import SimpleITK as sitk
import numpy as np
import os
import argparse
from tqdm import tqdm


sitk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(16)

def normalize_image(image):
    """Normalize image pixel values between 0 and 1."""
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)


class DVFCalculator:
    def __init__(self, sampling_percentage, shrink_factors, smoothing_sigmas, histogram_bins):
        self.__sampling_pecentage = sampling_percentage
        self.__shrink_factors = shrink_factors
        self.__smoothing_sigmas = smoothing_sigmas
        self.__histogram_bins = histogram_bins

    def compute(self, fixed_image, moving_image, initial_transform=None):
        """
        Compute the 3D DVF between two images.
        
        Args:
        - fixed_image: The image that will be used as the reference.
        - moving_image: The image that will be aligned to the fixed_image.
        - initial_transform: Initial transformation to be applied, if any.
        
        Returns:
        - dvf: The displacement vector field as a numpy array.
        """
        # Define the registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings: Using Mattes Mutual Information
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=self.__histogram_bins)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(self.__sampling_pecentage)

        # Multi-resolution settings
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=self.__shrink_factors)
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=self.__smoothing_sigmas)
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Optimizer settings
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Set initial transform using BSpline
        bspline_transform = sitk.BSplineTransformInitializer(fixed_image, [4,4,4])
        if initial_transform:
            bspline_transform.AddTransform(initial_transform)
        registration_method.SetInitialTransform(bspline_transform, inPlace=False)

        # Register the images
        final_transform = registration_method.Execute(fixed_image, moving_image)

        # Compute the DVF
        dvf = sitk.TransformToDisplacementField(final_transform, 
                                                outputPixelType=sitk.sitkVectorFloat64,
                                                size=fixed_image.GetSize(),
                                                outputOrigin=fixed_image.GetOrigin(),
                                                outputSpacing=fixed_image.GetSpacing(),
                                                outputDirection=fixed_image.GetDirection())
        return sitk.GetArrayFromImage(dvf), final_transform  # Return transform for potential use in the next timestep


def main(directory, sampling_percentage, shrink_factors, smoothing_sigmas, histogram_bins):
    dvf_calculator = DVFCalculator(sampling_percentage, shrink_factors, smoothing_sigmas, histogram_bins)

    # Recursively search for npz files
    npz_files = [os.path.join(root, file) 
                 for root, _, files in os.walk(directory)
                 for file in files if file.endswith(".npz")]

    for filepath in tqdm(npz_files, desc="Processing files"):
        # Load npz file
        data = np.load(filepath)['arr_0']
        data = normalize_image(data)

        dvfs = []
        initial_transform = None  # Use this to store the transform from the previous timestep
        for t in range(9):
            fixed_img_array = data[t]
            moving_img_array = data[t + 1]

            fixed_img = sitk.GetImageFromArray(fixed_img_array)
            moving_img = sitk.GetImageFromArray(moving_img_array)

            dvf, transform = dvf_calculator.compute(fixed_img, moving_img, initial_transform=initial_transform)
            dvfs.append(dvf)
            initial_transform = transform  # Update the transform

        dvfs = np.stack(dvfs, axis=0)

        # Save the DVF as a npz file
        save_path = os.path.join(os.path.dirname(filepath), os.path.basename(filepath).replace('.npz', '_dvf.npz'))
        np.savez(save_path, arr_0=dvfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute DVFs for 4D-CT data in a directory.")
    parser.add_argument("directory", type=str, help="Directory containing the 4D-CT npz files.")
    parser.add_argument("--sampling-percentage", type=float, default=0.01, 
                    help="Metric sampling percentage for registration. Default is 0.01.")
    parser.add_argument("--shrink-factors", type=int, nargs='+', default=[8, 4, 2, 1],
                        help="Shrink factors for multi-resolution approach.")
    parser.add_argument("--smoothing-sigmas", type=int, nargs='+', default=[4, 2, 1, 0],
                        help="Smoothing sigmas for multi-resolution approach.")
    parser.add_argument("--histogram-bins", type=int, default=30,
                        help="Number of histogram bins. Defualt is 30.")
    
    args = parser.parse_args()
    main(args.directory, args.sampling_percentage, args.shrink_factors, args.smoothing_sigmas, args.histogram_bins)
