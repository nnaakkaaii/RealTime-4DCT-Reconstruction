import SimpleITK as sitk
import numpy as np
import os
import argparse


def compute_dvf(fixed_image, moving_image):
    """
    Compute the 3D DVF between two images.
    
    Args:
    - fixed_image: The image that will be used as the reference.
    - moving_image: The image that will be aligned to the fixed_image.
    
    Returns:
    - dvf: The displacement vector field.
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
                                            sitk.VectorFloat64, 
                                            fixed_image.GetSize(),
                                            fixed_image.GetOrigin(),
                                            fixed_image.GetSpacing(),
                                            fixed_image.GetDirection())
    return dvf

def main(directory):
    # Recursively search for npz files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npz'):
                filepath = os.path.join(root, file)
                # Load npz file
                data = np.load(filepath)['arr_0']
                
                dvfs = []
                for t in range(9):
                    fixed_img_array = data[t]
                    moving_img_array = data[t + 1]
                    
                    fixed_img = sitk.GetImageFromArray(fixed_img_array)
                    moving_img = sitk.GetImageFromArray(moving_img_array)
                    
                    dvf = compute_dvf(fixed_img, moving_img)
                    dvfs.append(sitk.GetArrayFromImage(dvf))
                
                dvfs = np.stack(dvfs, axis=0)
                
                # Save the DVF as a npz file
                save_path = os.path.join(root, file.replace('.npz', '_dvf.npz'))
                np.savez(save_path, arr_0=dvfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute DVFs for 4D-CT data in a directory.")
    parser.add_argument("directory", type=str, help="Directory containing the 4D-CT npz files.")
    args = parser.parse_args()
    main(args.directory)
