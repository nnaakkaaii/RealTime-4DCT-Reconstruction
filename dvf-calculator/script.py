import SimpleITK as sitk
import numpy as np
import os
import argparse
from tqdm import tqdm


class DVFCalculator:
    def __init__(self, shrink_factors, smoothing_sigmas):
        self.__shrink_factors = shrink_factors
        self.__smoothing_sigmas = smoothing_sigmas

    def compute(self, fixed_image, moving_image):
        """
        Compute the 3D DVF between two images.
        
        Args:
        - fixed_image: The image that will be used as the reference.
        - moving_image: The image that will be aligned to the fixed_image.
        
        Returns:
        - dvf: The displacement vector field as a numpy array.
        """
        # Define the registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)

        # Multi-resolution settings
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=self.__shrink_factors)
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=self.__smoothing_sigmas)
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Optimizer settings
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Set initial transform (identity)
        initial_transform = sitk.TranslationTransform(fixed_image.GetDimension())
        registration_method.SetInitialTransform(initial_transform, inPlace=True)

        # Register the images
        final_transform = registration_method.Execute(fixed_image, moving_image)

        # Compute the DVF
        dvf = sitk.TransformToDisplacementField(final_transform, 
                                                outputPixelType=sitk.sitkVectorFloat64,
                                                size=fixed_image.GetSize(),
                                                outputOrigin=fixed_image.GetOrigin(),
                                                outputSpacing=fixed_image.GetSpacing(),
                                                outputDirection=fixed_image.GetDirection())
        return sitk.GetArrayFromImage(dvf)


def main(directory, shrink_factors, smoothing_sigmas):
    dvf_calculator = DVFCalculator(shrink_factors, smoothing_sigmas)

    # Recursively search for npz files
    npz_files = [os.path.join(root, file) 
                 for root, dirs, files in os.walk(directory)
                 for file in files if file.endswith(".npz")]

    for filepath in tqdm(npz_files, desc="Processing files"):
        # Load npz file
        data = np.load(filepath)['arr_0']

        dvfs = []
        for t in range(9):
            fixed_img_array = data[t]
            moving_img_array = data[t + 1]

            fixed_img = sitk.GetImageFromArray(fixed_img_array)
            moving_img = sitk.GetImageFromArray(moving_img_array)

            dvf = dvf_calculator.compute(fixed_img, moving_img)
            dvfs.append(dvf)

        dvfs = np.stack(dvfs, axis=0)

        # Save the DVF as a npz file
        save_path = os.path.join(os.path.dirname(filepath), os.path.basename(filepath).replace('.npz', '_dvf.npz'))
        np.savez(save_path, arr_0=dvfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute DVFs for 4D-CT data in a directory.")
    parser.add_argument("directory", type=str, help="Directory containing the 4D-CT npz files.")
    parser.add_argument("--shrink-factors", type=int, nargs='+', default=[8, 4, 2, 1],
                        help="Shrink factors for multi-resolution approach.")
    parser.add_argument("--smoothing-sigmas", type=int, nargs='+', default=[4, 2, 1, 0],
                        help="Smoothing sigmas for multi-resolution approach.")
    
    args = parser.parse_args()
    main(args.directory)
