#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vehicle.msg import VdispLine
from vehicle.msg import VanishingPoint

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import cv2

from matplotlib import pyplot as plt
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

last_right = list([0, 0]);
last_left = list([0, 0]);



CUDA_VANISHING_POINT_DETECTOR_FILENAME = 'vanishing_point_detector.cu'


def get_gabor_filter_kernel(theta, frequency, sigma_x, sigma_y, size):
    """Returns both the imaginary and real parts of a Gabor kernel.

    A Gabor kernel is a Gaussian kernel modulated by a complex harmonic
    function whose real and imaginary parts consist of a cosine and sine
    functions respectively.
    Harmonic function consists of an imaginary sine function and a real
    cosine function.

    Args:
      frequency : Frequency of the harmonic function in pixels.
      theta: Orientation in radians. If 0 is the x-direction.
      sigma_x: Standard deviation in x direction.
      sigma_y: Same as sigma_x but in the y direction. This applies to the
          kernel before rotation.
      size: Kernel size which applies to both dimensions.
    """
    y, x = np.mgrid[-int(size / 2):int(size / 2) + 1,
                    -int(size / 2):int(size / 2) + 1]
    
    rot_x = x * np.cos(theta) + y * np.sin(theta)
    rot_y = -x * np.sin(theta) + y * np.cos(theta)
    
    g = np.zeros(y.shape, dtype=np.complex)
    g[:] = np.exp(
        -0.5 * (rot_x ** 2 / sigma_x ** 2 + rot_y ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= np.exp(1j * (2 * np.pi * frequency * rot_x))

    return g.real.astype(np.float32), g.imag.astype(np.float32)


def get_gabor_filter_kernels(thetas, frequencies, sigma_x, sigma_y, size):
    """Returns an array of Gabor filter kernels loaded in GPU where position
       [i][j][k] means the kernel of ith theta, jth frequency, kth complex part
       (where 0 = real, 1 = imag)."""
    def get_gpu_loaded(kernel):
        kernel_gpu = cuda.mem_alloc(kernel.nbytes)
        cuda.memcpy_htod(kernel_gpu, kernel)
        return kernel_gpu
    return [[[get_gpu_loaded(kernel) for kernel in
              get_gabor_filter_kernel(theta, frequency, sigma_x, sigma_y, size)]
             for frequency in frequencies]
            for theta in thetas]


def apply_gabor_kernels(grey_image, gabor_kernels_gpu):
    """Applies all the bank of complex Gabor Kernels to calculate the Gabor
       Energies Tensor. The Gabor Energy Tensor contains at position
       [y, x, theta] the average over all frequencies for the magnitude response
       of convolutions at a given (x, y) pixel for a given theta. Note that the
       number of rows and columns in the Gabor Energies Tensor is
       (image_rows - (kernel_size >> 1)) X (image_cols - (kernel_size >> 1)) due
       to the padding lost at convolution."""
    
    original_rows, original_cols = grey_image.shape
    energies_rows = original_rows - GABOR_FILTER_KERNEL_SIZE + 1
    energies_cols = original_cols - GABOR_FILTER_KERNEL_SIZE + 1

    grey_image_gpu = cuda.mem_alloc(grey_image.nbytes)
    cuda.memcpy_htod(grey_image_gpu, grey_image)

    gabor_energies_gpu = cuda.mem_alloc(
        np.float32().itemsize * energies_rows * energies_cols * THETA_N)

    resetGaborEnergiesTensor(
        gabor_energies_gpu, np.int32(original_rows), np.int32(original_cols),
        np.int32(GABOR_FILTER_KERNEL_SIZE),
        block=(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1),
        grid=(int(np.ceil(original_cols / CUDA_BLOCK_SIZE)),
              int(np.ceil(original_rows / CUDA_BLOCK_SIZE))))
    for theta_idx in range(THETA_N):
        for freq_idx in range(FREQUENCIES_N):
            addGaborFilterMagnitudeResponse(
                gabor_energies_gpu, np.int32(theta_idx),
                grey_image_gpu, np.int32(original_rows), np.int32(original_cols),
                gabor_kernels_gpu[theta_idx][freq_idx][0],
                gabor_kernels_gpu[theta_idx][freq_idx][1],
                np.int32(GABOR_FILTER_KERNEL_SIZE),
                block=(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1),
                grid=(int(np.ceil(original_cols / CUDA_BLOCK_SIZE)),
                      int(np.ceil(original_rows / CUDA_BLOCK_SIZE))))
    divideGaborEnergiesTensor(
        gabor_energies_gpu, np.int32(original_rows), np.int32(original_cols),
        np.int32(GABOR_FILTER_KERNEL_SIZE), np.int32(FREQUENCIES_N),
        block=(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1),
        grid=(int(np.ceil(original_cols / CUDA_BLOCK_SIZE)),
              int(np.ceil(original_rows / CUDA_BLOCK_SIZE))))
    return gabor_energies_gpu


def get_vanishing_point_callback(color_image_msg,
                                 vdisp_line,
                                 cv_bridge,
                                 cuda_context,
                                 gabor_kernels_gpu,
                                 vanishing_point_pub,
                                 gabor_filter_kernels_pubs=None,
                                 gabor_energies_pubs=None,
                                 gabor_combined_energies_pub=None,
                                 vp_candidates_region_pub=None):
    """Publishes the vanishing_point by processing the camera's grey_image,
    exploiting the previously obtained vdisp_line.

    Args:
      color_image_msg: A ROS Message containing the road color Image.
          (TODO: Change this to the grey_image_msg once in recordings)
      vdisp_line: A ROS Message containing the vdisparity fitted line.
      cv_bridge: The CV bridge instance used to convert CV images and ROS
          Messages.
      cuda_context: The CUDA context to be used to execute kernels.
      vanishing_point_pub: The ROS Publisher for the vanishing_point.
      gabor_filter_kernels_pubs: If set, it's an array of ROS Publishers where
          each Publisher matches a theta of the Gabor Kernels. The corresponding
          kernel will be published taking into consideration whether the real
          or imaginary parts are desired and which frequency is to be published
          based on the param configuration.
      gabor_energies_pubs: If set, it's an array of ROS Publishers where each
          Publisher matches a theta of the applied Gabor kernels.
      gabor_combined_energies_pub: If set, it's a ROS Publisher where the
          magnitude response after processing the Gabor Energies Tensor into a
          single combined response per pixel.
      vp_candidates_region_pub: If set, it's a ROS Publisher where the resulting
          vp_candidates_voting_region after voting will be placed.
    """
    color_image = cv_bridge.imgmsg_to_cv2(
        color_image_msg, desired_encoding='passthrough')
    grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    
    
    grey_image = grey_image[int(np.round(vdisp_line.b)) 
				- int(GABOR_FILTER_KERNEL_SIZE / 2):, :]
    grey_image = cv2.resize(grey_image, (0,0),fx = SHRINK_FACTOR, fy = SHRINK_FACTOR)
    
    kernel = np.ones((3, 3), dtype = np.float32)
    kernel *= -1
    kernel[1][1] = 9
    
    grey_image = cv2.GaussianBlur(grey_image, (3, 3), 0);
    grey_image = cv2.filter2D(grey_image,-1, kernel)
    
    cuda_context.push()
    
    original_rows, original_cols = grey_image.shape
    
    energies_rows = original_rows - GABOR_FILTER_KERNEL_SIZE + 1
    energies_cols = original_cols - GABOR_FILTER_KERNEL_SIZE + 1

    energies_gpu = apply_gabor_kernels(grey_image, gabor_kernels_gpu)


    
    combined_energies_gpu = cuda.mem_alloc(
        np.float32().itemsize * energies_rows * energies_cols) 
    combined_phases_gpu = cuda.mem_alloc(
        np.float32().itemsize * energies_rows * energies_cols)
    confidence_gpu = cuda.mem_alloc(
        np.float32().itemsize * energies_rows * energies_cols
    )
    combineGaborEnergies(
        energies_gpu, np.int32(energies_rows), np.int32(energies_cols),
        combined_energies_gpu, combined_phases_gpu,  confidence_gpu,
        block=(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1),
        grid=(int(np.ceil(energies_cols / CUDA_BLOCK_SIZE)),
              int(np.ceil(energies_rows / CUDA_BLOCK_SIZE))))
    
    confidence = np.empty((energies_rows, energies_cols), dtype = np.float32)
    cuda.memcpy_dtoh(confidence, confidence_gpu)


    phases = np.empty((energies_rows, energies_cols),
                                     dtype=np.float32)
    cuda.memcpy_dtoh(phases, combined_phases_gpu);
    
    #for x in range(100, 1000, 50):
    #    for y in range(350, 500, 50):
    
    vp_candidates_gpu = cuda.mem_alloc(np.float32().itemsize *
                                      (2 * VP_HORIZON_CANDIDATES_MARGIN + 1) *
                                       energies_cols)
    cuda.memcpy_htod(vp_candidates_gpu, np.zeros((2 * VP_HORIZON_CANDIDATES_MARGIN + 1 , energies_cols), dtype=np.float32))


    
    voteForVanishingPointCandidates(
        combined_energies_gpu, combined_phases_gpu, vp_candidates_gpu,
        np.int32(energies_rows), np.int32(2 * VP_HORIZON_CANDIDATES_MARGIN + 1), np.int32(energies_cols),
        block=(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1),
        grid=(int(np.ceil(energies_cols / CUDA_BLOCK_SIZE)),
              int(np.ceil(energies_rows / CUDA_BLOCK_SIZE))))

    
    vp_candidates = np.empty((2*VP_HORIZON_CANDIDATES_MARGIN +1
				,energies_cols), dtype = np.float32)

    
    cuda.memcpy_dtoh(vp_candidates, vp_candidates_gpu)

    vp_candidates = cv2.GaussianBlur(vp_candidates,  (15, 15), 0)
    vp_candidates -= vp_candidates.min()
    vp_candidates /= vp_candidates.max()
    vp_candidates *=255
    
    
   
    #max
    vp = np.where(vp_candidates == np.amax(vp_candidates));
  
    
    vp_col = vp[1][0] ;
    
    
    max_rows = (color_image.shape[0])*SHRINK_FACTOR
    
    #vp_row = int(np.round(vdisp_line.b))*SHRINK_FACTOR
    #vp_row = (max_rows-(2*VP_HORIZON_CANDIDATES_MARGIN-vp[0][0]+GABOR_FILTER_KERNEL_SIZE/2))
    vp_row = 2*VP_HORIZON_CANDIDATES_MARGIN +1 - vp[0][0];
    #vp_score = [0, 0];
    
    VP_CANDIDATE_NUMBER = 7**2

    direction_array_gpu = cuda.mem_alloc(np.float32().itemsize*180*VP_CANDIDATE_NUMBER)
    candidates_array_gpu = cuda.mem_alloc(np.int32().itemsize*VP_CANDIDATE_NUMBER*2)
    vp_score_gpu = cuda.mem_alloc(np.int32().itemsize*VP_CANDIDATE_NUMBER);
    
    
    direction_array = np.zeros((VP_CANDIDATE_NUMBER, 180), dtype = np.float32);
    candidates_array = np.zeros((VP_CANDIDATE_NUMBER, 2), dtype = np.int32)
    vp_score = np.zeros(VP_CANDIDATE_NUMBER, dtype = np.float32)

    
    for i in range (VP_CANDIDATE_NUMBER):
        candidates_array[i][0] = energies_rows- vp_row +(i % np.int32(np.sqrt(VP_CANDIDATE_NUMBER)))*5-np.int32(np.sqrt(VP_CANDIDATE_NUMBER))*2 #+energies_rows-max_rows+GABOR_FILTER_KERNEL_SIZE/2
        candidates_array[i][1] = vp_col - (i / np.int32(np.sqrt(VP_CANDIDATE_NUMBER)))*5+np.int32(np.sqrt(VP_CANDIDATE_NUMBER))*2
    
    
    row = vp_row -max_rows+GABOR_FILTER_KERNEL_SIZE/2 +energies_rows;

       


    cuda.memcpy_htod(direction_array_gpu, direction_array)
    cuda.memcpy_htod(candidates_array_gpu, candidates_array)
    cuda.memcpy_htod(vp_score_gpu, vp_score)
    
    getRoadEdges(np.int32(VP_CANDIDATE_NUMBER), combined_energies_gpu,
     combined_phases_gpu, np.int32(energies_rows), np.int32(energies_cols), candidates_array_gpu, direction_array_gpu, vp_score_gpu,
   
        block=(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1),
        grid=(int(np.ceil(energies_cols / CUDA_BLOCK_SIZE)),
              int(np.ceil(energies_rows / CUDA_BLOCK_SIZE))))


    

    cuda.memcpy_dtoh(direction_array, direction_array_gpu)
    
   
    cuda.memcpy_dtoh(vp_score, vp_score_gpu)
    

    
    vp_score -=vp_score.min()
    vp_score /=vp_score.max()
    vp_score *= 255;

    
    vp_max = np.where(vp_score ==np.max(vp_score))[0]
    
    
    vp_row = candidates_array[vp_max[0]][0]-energies_rows+max_rows-GABOR_FILTER_KERNEL_SIZE
    vp_col = candidates_array[vp_max[0]][1] +GABOR_FILTER_KERNEL_SIZE/2
    
    row = candidates_array[vp_max[0]][0]

    supporting_pixels_gpu = cuda.mem_alloc(np.float32().itemsize*energies_rows*energies_cols)

    getSupportingPixels(np.int32(row), np.int32(vp_col), combined_phases_gpu, combined_energies_gpu, np.int32(energies_rows), np.int32(energies_cols), supporting_pixels_gpu,
    block=(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1),
        grid=(int(np.ceil(energies_cols / CUDA_BLOCK_SIZE)),
              int(np.ceil(energies_rows / CUDA_BLOCK_SIZE)))
    
    )

    supporting_pixels = np.zeros((energies_rows, energies_cols), dtype=np.float32)
    cuda.memcpy_dtoh(supporting_pixels, supporting_pixels_gpu)

    vanishing_point_pub.publish(  # TODO: Publish real VP
        VanishingPoint(header=color_image_msg.header, row=vp_row/SHRINK_FACTOR, col=(vp_col)*1/SHRINK_FACTOR))

    supporting_pixels -= supporting_pixels.min();
    supporting_pixels /= supporting_pixels.max();
    supporting_pixels *= 255;
    
    direction_vector = direction_array[vp_max[0]]
    
    direction_vector -= direction_vector.min();
    direction_vector /= direction_vector.max();
    direction_vector *= 255;

    hist = np.ones((300*2, 180*4), np.uint8)
    hist *= 255;
    max_left =0;
    max_right = 0;
    index_right = 0;
    index_left = 0;
    vp_score = np.int32(vp_score)
    global last_left;
    global last_right;
    
    
    for i in range (0, 180):
        
        for j in range (0, np.uint8(direction_vector[i])):
            if i % 30 !=0:
                hist[(299-j)*2][i*4] = 0;
                hist[(299-j)*2][i*4+1] = 0;
                hist[(299-j)*2][i*4+2] = 0;
                hist[(299-j)*2][i*4+3] = 0;
        
    

    for i in range (0, 90):
        if direction_vector[i]>max_left:
            
            index_left = i
            max_left = direction_vector[index_left]
            
        if direction_vector[179-i]>max_right:
            
            index_right = 179-i
            max_right = direction_vector[index_right];
            
    
  
    x = 750
    y = 450
    #cv2.line(color_image, (x, y), (np.int(vp_col/SHRINK_FACTOR), np.int(vp_row/SHRINK_FACTOR)), (0, 0, 255), 2)   
    #cv2.arrowedLine(color_image, (x, y), (np.int32(x+20*np.cos(phases[np.int((y-544)*SHRINK_FACTOR+102+GABOR_FILTER_KERNEL_SIZE/2)][np.int(x*SHRINK_FACTOR-GABOR_FILTER_KERNEL_SIZE/2)])),
                      #                          np.int32(y-20*np.sin(phases[np.int((y-544)*SHRINK_FACTOR+102+GABOR_FILTER_KERNEL_SIZE/2)][np.int(x*SHRINK_FACTOR-GABOR_FILTER_KERNEL_SIZE/2)]) )), (255, 0, 0), 2)
    
    
    

    index_left = (last_left[0]+last_left[1]+index_left)/3;
    last_left[0] = last_left[1]
    last_left[1] = index_left;

   
    
    index_right = (last_right[0]+last_right[1]+index_right)/3;
    last_right[0] = last_right[1]
    last_right[1] = index_right;

    """
    for i in range(180):
        cv2.line(color_image, (np.int32(vp_col/SHRINK_FACTOR), np.int32((vp_row/SHRINK_FACTOR))),
        (np.int32(vp_col/SHRINK_FACTOR-200*np.cos(i*np.pi/180)),
        np.int32(vp_row/SHRINK_FACTOR + 200*np.sin(i*np.pi/180))), 
        (np.int(direction_vector[i]), 0, 255-np.int(direction_vector[i])), 1)
    """
    #cv2.imshow("phases", color_image);
    cv2.line(color_image, (np.int32(vp_col/SHRINK_FACTOR), np.int32((vp_row/SHRINK_FACTOR))),
        (np.int32(vp_col/SHRINK_FACTOR-600*np.cos(index_left*np.pi/180)),
        np.int32(vp_row/SHRINK_FACTOR + 600*np.sin(index_left*np.pi/180))), 
        (0, 255, 0), 5)
    
    cv2.line(color_image, (np.int32(vp_col/SHRINK_FACTOR), np.int32(vp_row/SHRINK_FACTOR)),
            (np.int32(vp_col/SHRINK_FACTOR-600*np.cos(index_right*np.pi/180)),
            np.int32(vp_row/SHRINK_FACTOR + 600*np.sin(index_right*np.pi/180))), 
            (0, 255, 0), 5)
    cv2.imshow("image",color_image)
    cv2.waitKey(3);
    
    

    if gabor_filter_kernels_pubs is not None:
        for theta_idx in range(THETA_N):
            kernel = np.empty(
                (GABOR_FILTER_KERNEL_SIZE, GABOR_FILTER_KERNEL_SIZE),
                dtype=np.float32)
            cuda.memcpy_dtoh(
                kernel,
                gabor_kernels_gpu[theta_idx]
                                 [PUBLISH_GABOR_FILTERS_FREQUENCY_IDX]
                                 [PUBLISH_GABOR_FILTERS_IMAG_PART])
            kernel -= kernel.min()
            kernel /= kernel.max()
            kernel *= 255
            gabor_filter_kernels_pubs[theta_idx].publish(cv_bridge.cv2_to_imgmsg(
                kernel.astype(np.uint8), encoding='8UC1'))

    if gabor_energies_pubs is not None:
        energies = np.empty(
            (energies_rows, energies_cols, THETA_N), dtype=np.float32)
        cuda.memcpy_dtoh(energies, energies_gpu)
        for theta_idx in range(THETA_N):
            energy = energies[:, :, theta_idx]
            energy -= energy.min()
            energy /= energy.max()
            energy *= 255
            gabor_energies_pubs[theta_idx].publish(cv_bridge.cv2_to_imgmsg(
                energy.astype(np.uint8), encoding='8UC1'))

    if gabor_combined_energies_pub is not None:
        combined_energies = np.empty((energies_rows, energies_cols),
                                     dtype=np.float32)
        cuda.memcpy_dtoh(combined_energies, combined_energies_gpu)
	
	combined_energies = np.nan_to_num(combined_energies)
	combined_energies[combined_energies==np.NINF] = 0
	combined_energies[combined_energies==np.inf] = 0
        combined_energies -= combined_energies.min()
	if combined_energies.max()!=0:
        	combined_energies /= combined_energies.max()
        combined_energies *= 255
	
	
	
        gabor_combined_energies_pub.publish(cv_bridge.cv2_to_imgmsg(
            #combined_energies.astype(np.uint8)
            #hist
            supporting_pixels.astype(np.uint8)
            , encoding='8UC1'))
            
	

    if vp_candidates_region_pub is not None:
        vp_candidates_region = np.empty(
            (2 * VP_HORIZON_CANDIDATES_MARGIN + 1, energies_cols),
            dtype=np.float32)
        cuda.memcpy_dtoh(vp_candidates_region, vp_candidates_gpu)
        vp_candidates_region = cv2.GaussianBlur(vp_candidates_region, (15, 15), 0)
        max_idx = np.unravel_index(np.argmax(vp_candidates_region),
                                   vp_candidates_region.shape)
        vp_candidates_region -= vp_candidates_region.min()
        vp_candidates_region /= vp_candidates_region.max()
        vp_candidates_region *= 255
	
        vp_candidates_region[vp[0][0]][ vp[1][0]] = 0;
        
        vp_candidates_region_pub.publish(cv_bridge.cv2_to_imgmsg(
            #grey_image, 
            vp_candidates_region.astype(np.uint8),
            
            #confidence.astype(np.uint8),
            encoding='8UC1'))

    
		   


if __name__ == '__main__':
    vpGraph=[];
    plt.ion();
    rospy.init_node('vanishing_point_detector', anonymous=False)
    
    # The following publications are for debugging and visualization only as
    # they severely slow down node execution.
    PUBLISH_GABOR_FILTER_KERNELS = rospy.get_param(
        '~publish_gabor_filter_kernels', False)
    PUBLISH_GABOR_FILTERS_IMAG_PART = rospy.get_param(
        '~publish_gabor_filter_imag_part', True)
    PUBLISH_GABOR_FILTERS_FREQUENCY_IDX = rospy.get_param(
        '~publish_gabor_filter_frequency_idx', 0)
    PUBLISH_GABOR_ENERGIES = rospy.get_param(
        '~publish_gabor_energies', False)
    PUBLISH_COMBINED_GABOR_ENERGIES = rospy.get_param(
        '~publish_combined_gabor_energies', False)
    PUBLISH_VP_CANDIDATES_VOTING_REGION = rospy.get_param(
        '~publish_vp_candidates_voting_region', False)

    # Vanishing Point Detection Parameters
    CUDA_BLOCK_SIZE = rospy.get_param('~cuda_block_size', 16)
    VP_HORIZON_CANDIDATES_MARGIN = rospy.get_param(
        '~vp_horizon_candidates_margin', 100)
    THETA_N = 4  # Implementation Specific
    GABOR_FILTER_THETAS = [
        np.pi / THETA_N * theta_idx for theta_idx in range(THETA_N)]
    GABOR_FILTER_KERNEL_SIZE = rospy.get_param(
        '~gabor_filter_kernel_size', 25)  # Should be odd.
    assert GABOR_FILTER_KERNEL_SIZE % 2 == 1
    GABOR_FILTER_SIGMA_X = rospy.get_param('~gabor_filter_sigma_x', 4)
    GABOR_FILTER_SIGMA_Y = rospy.get_param('~gabor_filter_sigma_y', 2)
    # Recommendation: Values between 0.1 and 0.5
    GABOR_FILTER_FREQUENCIES = rospy.get_param(
        '~gabor_filter_frequencies', [0.2])
    FREQUENCIES_N = len(GABOR_FILTER_FREQUENCIES)

    SHRINK_FACTOR = .5; #1 is original image

    cuda.init()
    cuda_device = cuda.Device(0)
    cuda_context = cuda_device.make_context()
    with open(CUDA_VANISHING_POINT_DETECTOR_FILENAME, 'r') as f:
        mod = SourceModule(f.read(), no_extern_c=True)
    resetGaborEnergiesTensor = mod.get_function('resetGaborEnergiesTensor')
    addGaborFilterMagnitudeResponse = mod.get_function('addGaborFilterMagnitudeResponse')
    divideGaborEnergiesTensor = mod.get_function('divideGaborEnergiesTensor')
    combineGaborEnergies = mod.get_function('combineGaborEnergies')
    voteForVanishingPointCandidates = mod.get_function('voteForVanishingPointCandidates')
    getRoadEdges = mod.get_function('getRoadEdges')
    getSupportingPixels = mod.get_function('getSupportingPixels')

    cv_bridge = CvBridge()
   
   
    vanishing_point_pub = rospy.Publisher(
        'vanishing_point', VanishingPoint, queue_size=1)
    gabor_filter_kernels_pubs = (
        [rospy.Publisher('gabor_%d_filter_kernel' % (180 / THETA_N * theta_idx),
                         Image,
                         queue_size=1)
         for theta_idx in range(THETA_N)]
        if PUBLISH_GABOR_FILTER_KERNELS else None)
    gabor_energies_pubs = (
        [rospy.Publisher('gabor_%d_energy' % (180 / THETA_N * theta_idx),
                         Image,
                         queue_size=1)
         for theta_idx in range(THETA_N)]
        if PUBLISH_GABOR_ENERGIES else None)
    gabor_combined_energies_pub = (
        rospy.Publisher('gabor_combined_energies', Image, queue_size=1)
        if PUBLISH_COMBINED_GABOR_ENERGIES else None)
    vp_candidates_region_pub = (
        rospy.Publisher('vp_candidates_region', Image, queue_size=1)
        if PUBLISH_VP_CANDIDATES_VOTING_REGION else None)

    color_image_sub = message_filters.Subscriber(
        '/multisense/left/image_rect_color', Image)
    vdisp_image_sub = message_filters.Subscriber(
        '/vdisp_line_fitter/vdisp_line', VdispLine)
    ts = message_filters.TimeSynchronizer(
        [color_image_sub, vdisp_image_sub], queue_size=5)
    ts.registerCallback(get_vanishing_point_callback,
                        cv_bridge,
                        cuda_context,
                        get_gabor_filter_kernels(GABOR_FILTER_THETAS,
                                                 GABOR_FILTER_FREQUENCIES,
                                                 GABOR_FILTER_SIGMA_X,
                                                 GABOR_FILTER_SIGMA_Y,
                                                 GABOR_FILTER_KERNEL_SIZE),
                        vanishing_point_pub,
                        gabor_filter_kernels_pubs,
                        gabor_energies_pubs,
                        gabor_combined_energies_pub,
                        vp_candidates_region_pub)
    plt.show(block=True)
    rospy.spin()
    