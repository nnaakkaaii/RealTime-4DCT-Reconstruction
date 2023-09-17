import SimpleITK as sitk
import numpy as np
import os
import argparse
from tqdm import tqdm


def compute_dvf(fixed_image, moving_image):
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
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
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

def main(directory):
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

            dvf = compute_dvf(fixed_img, moving_img)
            dvfs.append(dvf)

        dvfs = np.stack(dvfs, axis=0)

        # Save the DVF as a npz file
        save_path = os.path.join(os.path.dirname(filepath), os.path.basename(filepath).replace('.npz', '_dvf.npz'))
        np.savez(save_path, arr_0=dvfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute DVFs for 4D-CT data in a directory.")
    parser.add_argument("directory", type=str, help="Directory containing the 4D-CT npz files.")
    args = parser.parse_args()
    main(args.directory)
