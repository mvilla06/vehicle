#define THETA_N 4
#define SQRT_2 1.4142135623730951f
#define PI 3.141592653589793f

extern "C" {
/**
 * Clears out the Gabor Energies Tensor, setting all of its values to zero.
 * The Gabor Energies Tensor is the data structure whose [y, x, theta] value contains the average magnitude response to
 * the different complex 'Gabor' filters for an specific 'theta' orientation at 'image' location (y, x).
 * This is the first step towards its calculation. Note that the number of rows and columns in the Gabor Energies Tensor
 * is (image_rows - (kernel_size >> 1)) X (image_cols - (kernel_size >> 1)) due to the padding lost at convolution.
 * @param gabor_energies The Gabor Energies Tensor.
 * @param rows The number of rows in the 'image' whose Energies Tensor will be calculated.
 * @param cols The number of columns in the 'image' whose Energies Tensor will be calculated.
 * @param kernel_size Both the number of rows and columns in the Gabor kernels to apply. Should be an odd number.
 */
__global__ void resetGaborEnergiesTensor(float* gabor_energies, int rows, int cols, int kernel_size)
{
  int image_y = blockDim.y * blockIdx.y + threadIdx.y;
  int image_x = blockDim.x * blockIdx.x + threadIdx.x;
  int image_padding = (kernel_size >> 1);
  if (image_y < image_padding || image_y + image_padding >= rows ||
      image_x < image_padding || image_x + image_padding >= cols)
    return;  // Part of the padding lost due to lack of border information.
  int tensor_offset = ((image_y - image_padding) * (cols - (image_padding << 1)) + (image_x - image_padding)) * THETA_N;
  for (int i = 0; i < THETA_N; i++)
    gabor_energies[tensor_offset + i] = 0.0f;
}

/**
 * Applies a 2D Complex Convolution on a real image given a square kernel and adds its magnitude response to the
 * corresponding [y, x, theta] location in the Gabor Energies Tensor.
 * This kernel is the second step towards its calculation and should be called once for every frequency to apply.
 * @param gabor_energies The Gabor Energies Tensor.
 * @param theta_idx The orientation index for the Gabor Energies Tensor specifying the orientation for which to add
 *                  this convolution.
 * @param image The image on which to apply the convolution operation.
 * @param rows The number of rows in 'image'.
 * @param cols The number of columns in 'image'.
 * @param real_kernel The real part of the square convolution kernel to apply on 'image'.
 * @param imag_kernel The imaginary part of the square convolution kernel to apply on 'image'.
 * @param kernel_size Both the number of rows and columns in 'kernel'. Should be an odd number.
 */
__global__ void addGaborFilterMagnitudeResponse(float* gabor_energies, int theta_idx,
                                                unsigned char* image, int rows, int cols,
                                                float* real_kernel, float* imag_kernel, int kernel_size)
{
  int image_y = blockDim.y * blockIdx.y + threadIdx.y;
  int image_x = blockDim.x * blockIdx.x + threadIdx.x;
  int image_padding = (kernel_size >> 1);
  if (image_y < image_padding || image_y + image_padding >= rows ||
      image_x < image_padding || image_x + image_padding >= cols)
    return;  // Part of the padding lost due to lack of border information.
  int image_idx = (image_y - image_padding) * cols + (image_x - image_padding), kernel_idx = 0;
  float real_response = 0.0f, imag_response = 0.0f;
  for (int i = 0; i < kernel_size; i++)
  {
    for (int j = 0; j < kernel_size; j++)
    {
      real_response += image[image_idx] * real_kernel[kernel_idx];
      imag_response += image[image_idx] * imag_kernel[kernel_idx];
      image_idx++;
      kernel_idx++;
    }
    image_idx += cols - kernel_size;
  }
  int tensor_offset = ((image_y - image_padding) * (cols - (image_padding << 1)) + (image_x - image_padding)) * THETA_N;
  gabor_energies[tensor_offset + theta_idx] = sqrtf(real_response * real_response + imag_response * imag_response);
}

/**
 * Divides all of the Gabor Energies Tensor elements by a constant.
 * This is the third and last step to calculate the Tensor. This step is used to average out the magnitude responses of
 * the different applied Gabor kernels: for a given [y, x, theta], one is applied per frequency.
 * @param gabor_energies The Gabor Energies Tensor.
 * @param rows The number of rows in the 'image' whose Energies Tensor will be calculated.
 * @param cols The number of columns in the 'image' whose Energies Tensor will be calculated.
 * @param kernel_size Both the number of rows and columns in the applied Gabor kernels. Should be an odd number.
 * @param constant The number by which to divide all of the Gabor Energies Tensor elements. Should be equal to the
 *                 number of applied frequencies.
 */
__global__ void divideGaborEnergiesTensor(float* gabor_energies, int rows, int cols, int kernel_size, int constant)
{
  int image_y = blockDim.y * blockIdx.y + threadIdx.y;
  int image_x = blockDim.x * blockIdx.x + threadIdx.x;
  int image_padding = (kernel_size >> 1);
  if (image_y < image_padding || image_y + image_padding >= rows ||
      image_x < image_padding || image_x + image_padding >= cols)
    return;  // Part of the padding lost due to lack of border information.
  int tensor_offset = ((image_y - image_padding) * (cols - (image_padding << 1)) + (image_x - image_padding)) * THETA_N;
  for (int i = 0; i < THETA_N; i++)
    gabor_energies[tensor_offset + i] /= constant;
}

/**
 * Combines the Gabor Energies Tensor into a Matrix by joining the magnitude response of the different thetas into a
 * single one with a corresponding combined energy and combined phase (angle). This takes into consideration the two
 * strongest orientations (thetas) and linearly joining their equivalent plane components. The two weakest components
 * are subtracted from the strongest ones since random textures tend to equally respond to different Gabor kernels.
 * @param gabor_energies The Gabor Energies Tensor.
 * @param rows The number of rows in 'gabor_energies'.
 * @param cols The number of columns in 'gabor_energies'.
 * @param combined_energies The resulting magnitude response from combining the Gabor energies at different thetas.
 * @param combined_phases The resulting phase response from combining the Gabor energies at different thetas.
 */
__global__ void combineGaborEnergies(float* gabor_energies, int rows, int cols,
                                     float* combined_energies, float* combined_phases)
{
  int image_y = blockDim.y * blockIdx.y + threadIdx.y;
  int image_x = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = image_y * cols + image_x;
  if (image_y >= rows || image_x >= cols)
    return;  // Out of image.
  int descending_energies_arg[THETA_N];
  float temp_energies[THETA_N];
  for (int i = 0; i < THETA_N; i++)
    temp_energies[i] = gabor_energies[THETA_N * offset + i];
  for (int i = 0; i < THETA_N; i++)
  {
    int max_idx = 0;
    float max_energy = temp_energies[0];
    for (int j = 1; j < THETA_N; j++)
      if (temp_energies[j] > max_energy)
      {
        max_idx = j;
        max_energy = temp_energies[j];
      }
    descending_energies_arg[i] = max_idx;
    temp_energies[max_idx] = -1.0f;
  }
//consider only relevant voters, where the confidence is over a 0.35 threshold, approximating an ideal response to a sine function

if((1 - ((gabor_energies[THETA_N * offset + descending_energies_arg[1]] +
	gabor_energies[THETA_N * offset + descending_energies_arg[2]] +
	gabor_energies[THETA_N * offset + descending_energies_arg[3]] )/
	(3*gabor_energies[THETA_N * offset + descending_energies_arg[0]])))<0.7){
		combined_energies[offset] =0;
		combined_phases[offset]= PI/2;
		return; //confidence is below threshold, there is not a well defined orientation 
	}

  float s1 = (gabor_energies[THETA_N * offset + descending_energies_arg[0]] );//-
  //            gabor_energies[THETA_N * offset + descending_energies_arg[3]]);
  float s2 = (gabor_energies[THETA_N * offset + descending_energies_arg[1]]);// -
   //           gabor_energies[THETA_N * offset + descending_energies_arg[2]]);
  int theta_idx1 = descending_energies_arg[0];
  int theta_idx2 = descending_energies_arg[1];
  float combined_y = 0.0f, combined_x = 0.0f;
  switch(theta_idx1)
  {
    case 0:
      if (theta_idx2 == 1)
      {
        combined_y = s1 + s2 / SQRT_2;
        combined_x = s2 / SQRT_2;
      }
      else if (theta_idx2 == 3)
      {
        combined_y = s1 + s2 / SQRT_2;
        combined_x = -s2 / SQRT_2;
      }
      break;
    case 1:
      if (theta_idx2 == 0)
      {
        combined_y = s1 / SQRT_2 + s2;
        combined_x = s1 / SQRT_2;
      }
      else if (theta_idx2 == 2)
      {
        combined_y = s1 / SQRT_2;
        combined_x = s1 / SQRT_2 + s2;
      }
      break;
    case 2:
      if (theta_idx2 == 1)
      {
        combined_y = s2 / SQRT_2;
        combined_x = s1 + s2 / SQRT_2;
      }
      else if (theta_idx2 == 3)
      {
        combined_y = s2 / SQRT_2;
        combined_x = -s1 - s2 / SQRT_2;
      }
      break;
    case 3:
      if (theta_idx2 == 0)
      {
        combined_y = s1 / SQRT_2 + s2;
        combined_x = -s1 / SQRT_2;
      }
      else if (theta_idx2 == 2)
      {
        combined_y = s1 / SQRT_2;
        combined_x = -s1 / SQRT_2 - s2;
      }
      break;
  }
  combined_energies[offset] = sqrtf(combined_y * combined_y + combined_x * combined_x);
  combined_phases[offset] = atan2f(combined_y, combined_x);
}

/**
 * Generates votes for all of the Vanishing Point candidates by allowing all of the voting region to assign a voting
 * weight for their preferred candidates. The candidate region is assumed to be directly above the voting region
 * (combined components) such that concatenated are part of a continuous region of the original image.
 * @param combined_energies The resulting magnitude response from combining the Gabor energies at different thetas.
 * @param combined_phases The resulting phase response from combining the Gabor energies at different thetas.
 * @param candidates The Vanishing Point candidates, being a region directly above the voting region which should also
 *                   correspond to a stripe around the horizon line.
 * @param voters_rows The number of rows in both 'combined_energies' and 'combined_phases'.
 * @param candidates_rows The number of rows in 'candidates'.
 * @param cols The number of columns in all three: 'combined_energies', 'combined_phases', and 'candidates'.
 */
__global__ void voteForVanishingPointCandidates(float* combined_energies, float* combined_phases, float* candidates,
                                                int voters_rows, int candidates_rows, int cols)
{
  int image_y = blockDim.y * blockIdx.y + threadIdx.y;
  int image_x = blockDim.x * blockIdx.x + threadIdx.x;
  if (image_y >= voters_rows || image_x >= cols)
    return;  // Out of image.
  int energies_offset = image_y * cols + image_x;
  int candidates_y_offset = candidates_rows * cols;
  float energy = combined_energies[energies_offset];
  if (energy < 0.85)
    return;  // Filter Noise
  float phase = combined_phases[energies_offset];
  float cot = 1.0f / tanf(phase);
  for (int candidates_y = candidates_rows - 1; candidates_y >= 0; candidates_y--)
  {
    candidates_y_offset -= cols;
    int y_delta = image_y + candidates_rows - candidates_y;
    int candidates_x = image_x + cot * y_delta;
    if (candidates_x >= 0 && candidates_x < cols)
      atomicAdd(&candidates[candidates_y_offset + candidates_x], 1);//energy);
  }
}
}
