"""
This module contains raw function definitions that can be either used directly or wrapped and then
assembled in pipelines
"""

import os

import numpy as np
from scipy import ndimage as ndi, stats
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.filters import median, rank, threshold_otsu
from skimage.morphology import disk, dilation, watershed, closing, skeletonize, medial_axis
from skimage.segmentation import random_walker

from imagepipe.tools.helpers import PipeArgError

from collections import defaultdict

def split_and_trim(path_prefix, main_root):
    """
    helper function for OS Path trimming routine that accounts for the trailing separator

    :param path_prefix: [str]
    :param main_root: [str]
    :return:[list]
    """
    trim_length = len(main_root)
    print os.sep, main_root[-1]
    if main_root[-1] != os.sep:
        trim_length += 1

    return path_prefix[trim_length:].split(os.sep)


def f_3d_stack_2d_filter(_3d_stack, _2d_filter):
    """
    helper function to apply 2d filter to a 3d image along the z axis

    :param _3d_stack:
    :param _2d_filter:
    :return:
    """
    new_stack = np.zeros_like(_3d_stack)
    new_stack[:, _2d_filter] = _3d_stack[:, _2d_filter]
    return new_stack


def f_2d_stack_2d_filter(_2d_stack, _2d_filter):
    """
    helper function to apply the 2d filter to a 2d image while creating an image copy

    :param _2d_stack:
    :param _2d_filter:
    :return:
    """
    new_stack = np.zeros_like(_2d_stack)
    new_stack[_2d_filter] = _2d_stack[_2d_filter]
    return new_stack


def gamma_stabilize(image, alpha_clean=5, floor_method='min'):
    """
    Normalizes the luma curve. floor intensity becomes 0 and max allowed by the bit number - 1

    :param image:
    :param alpha_clean: size of features that would be removed if surrounded by a majority of
    :param floor_method: ['min', '1q', '5p', 'median'] method of setting the floor intensity. 1q is first quartile, 1p is the first percentile
    :return:
    """
    bits = dtype2bits[image.dtype.name]
    if floor_method == 'min':
        inner_min = np.min(image)
    elif floor_method == '1q':
        inner_min = np.percentile(image, 25)
    elif floor_method == '5p':
        inner_min = np.percentile(image, 5)
    elif floor_method == 'median':
        inner_min = np.median(image)
    else:
        raise PipeArgError('floor_method can only be one of the three types: min, 1q, 5p or median')
    stabilized = (image - inner_min) / (float(2 ** bits) - inner_min)
    stabilized[stabilized < alpha_clean*np.median(stabilized)] = 0
    return stabilized


def smooth(image, smoothing_px=1.5):
    """
    Gaussian smoothing of the image

    :param image:
    :param smoothing_px:
    :return:
    """
    for i in range(0, image.shape[0]):
        image[i, :, :] = gaussian_filter(image[i, :, :],
                                         smoothing_px, mode='constant')
        image[image < 5 * np.mean(image)] = 0
    return image


def smooth_2d(image, smoothing_px=1.5):
    """
    Gaussian smoothing of a 2d image

    :param image:
    :param smoothing_px:
    :return:
    """
    # if np.shape(image) == (3,3):
    #     image = np.max(image, axis = 0)
    image = gaussian_filter(image, smoothing_px, mode='constant')
    image[image < 5 * np.mean(image)] = 0
    # dbg.max_projection_debug(np.max(image, axis=0))
    return image


def sum_projection(image):
    """
    Sum projection along z axis

    :param image:
    :return:
    """
    # dbg.max_projection_debug(np.max(image, axis=0))
    # dbg.sum_proj_debug(np.sum(image, axis=0))
    return np.sum(image, axis=0)


def max_projection(current_image):
    """
    Max projection along z axis

    :param current_image:
    :return:
    """
    # dbg.max_projection_debug(np.max(current_image, axis=0))

    return np.max(current_image, axis=0)


def random_walker_binarize(base_image, _dilation=0):
    """
    Improved random walker binarization based on the the scikits image library

    :param base_image:
    :param _dilation: if set to anything other than 0, would perform a morphological dilation using this parameter value as size
    :return: binary labels
    """
    gfp_clustering_markers = np.zeros(base_image.shape, dtype=np.uint8)

    # To try: add a grey area around the boundary between black and white

    gfp_clustering_markers[base_image > np.mean(base_image) * 2] = 2
    gfp_clustering_markers[base_image < np.mean(base_image) * 0.20] = 1

    binary_labels = random_walker(base_image, gfp_clustering_markers, beta=10, mode='bf') - 1

    if _dilation:
        selem = disk(_dilation)
        binary_labels = dilation(binary_labels, selem)

    return binary_labels


def robust_binarize(base_image, _dilation=0, heterogeity_size=10, feature_size=50):
    """
    Robust binarization algorithm based off random walker clustering

    :param base_image:
    :param _dilation: if set to anything other than 0, would perform a morphological dilation using this parameter value as size
    :param heterogeity_size: size of the feature (px) that the method will try to eliminate by smoothing
    :param feature_size: size of the feature (px) that the method will try to segment out
    :return: binary_labels
    """
    if np.percentile(base_image, 99) < 0.20:
        if np.percentile(base_image, 99) > 0:
            mult = 0.20 / np.percentile(base_image, 99)  # poissonean background assumptions
        else:
            mult = 1000. / np.sum(base_image)
        base_image = base_image * mult
        base_image[base_image > 1] = 1

    clustering_markers = np.zeros(base_image.shape, dtype=np.uint8)

    selem = disk(heterogeity_size)
    smooth = gaussian_filter(base_image, heterogeity_size, mode='constant')
    smooth_median = median(smooth, selem)
    uniform_median = median(base_image, selem)

    selem2 = disk(feature_size)
    local_otsu = rank.otsu(smooth_median, selem2)
    uniform_median_otsu = rank.otsu(uniform_median, selem2)

    clustering_markers[smooth_median < local_otsu * 0.9] = 1
    clustering_markers[smooth_median > local_otsu * 1.1] = 2

    # dbg.random_walker_debug(smooth_median, clustering_markers)
    # dbg.robust_binarize_debug(base_image, smooth_median, smooth_median, local_otsu, clustering_markers,
    #                           0, uniform_median, uniform_median_otsu)

    binary_labels = random_walker(smooth_median, clustering_markers, beta=10, mode='bf') - 1

    if _dilation:
        selem = disk(_dilation)
        binary_labels = dilation(binary_labels, selem)

    # dbg.robust_binarize_debug(binary_labels, base_image)
    # dbg.voronoi_debug(binary_labels, local_maxi, dist, segmented_cells_labels)
    # dbg.Kristen_robust_binarize(binary_labels, base_image)
    return binary_labels


def filter_labels(labels, binary_mask, min_feature_size=10):
    """
    Applies the binary mask to labels, than filters out all the labels with feature size less than
    min_feature_size

    :param labels:
    :param binary_mask:
    :param min_feature_size:
    :return:
    """
    binary_mask = binary_mask.astype(np.bool)

    filtered_labels = np.zeros_like(labels)
    filtered_labels[binary_mask] = labels[binary_mask]

    mask_items = np.unique(labels)
    mask_items = mask_items[mask_items > 0].tolist()

    for val in mask_items:
        _mask = filtered_labels == val
        if len(_mask):
            px_radius = np.sqrt(np.sum((filtered_labels == val).astype(np.int)))
        else:
            px_radius = 0
        if px_radius < min_feature_size:
            filtered_labels[labels == val] = labels[labels == val]

    # dbg.filter_labels_debug(labels, binary_mask, filtered_labels)

    return filtered_labels


def voronoi_segment_labels(binary_labels):
    """
    Performs a Voronoi segmentation on binary labels (assuming background is set to 0)

    :param binary_labels:
    :return: pad with labels of segmentation of the same size as binary_labels
    """

    dist = ndi.morphology.distance_transform_edt(np.logical_not(binary_labels))
    segmented_cells_labels = watershed(dist, binary_labels)

    return segmented_cells_labels


def exclude_region(exclusion_mask, base_image, _dilation=5):
    """
    Excludes the region where exclusion_mask is true from the base image.

    :param exclusion_mask:
    :param base_image:
    :param _dilation: if set to anything other than 0, would perform a morphological dilation using this parameter value as size on the exclusion mask
    :return:
    """
    _exclusion_mask = np.zeros_like(exclusion_mask)
    _exclusion_mask[exclusion_mask > 0] = 1

    if _dilation:
        selem = disk(_dilation)
        _exclusion_mask = dilation(_exclusion_mask, selem)

    excluded = np.zeros_like(base_image)
    excluded[np.logical_not(_exclusion_mask)] = base_image[np.logical_not(_exclusion_mask)]

    return excluded


def in_contact(mask1, mask2, distance=10):
    """
    Finds if two binary masks are in contact or proximity. distance of detection is defined by the distance parameter

    :param mask1:
    :param mask2:
    :param distance:
    :return: two arrays of the same shape as masks, each with ones for labels that overlap.
    """

    selem = disk(distance)

    extended_msk1 = dilation(mask1, selem)
    extended_msk2 = dilation(mask2, selem)

    intersection = np.logical_and(extended_msk1, extended_msk2)
    intersection = dilation(intersection, selem)

    labeled_mask1, tot1 = ndi.label(mask1)
    labeled_mask2, tot2 = ndi.label(mask2)

    in_contact1 = np.zeros_like(mask1)
    in_contact2 = np.zeros_like(mask2)

    for label in range(1, tot1+1):
        if np.any(intersection[labeled_mask1 == label]):
            in_contact1[labeled_mask1 == label] = 1

    for label in range(1, tot2+1):
        if np.any(intersection[labeled_mask2 == label]):
            in_contact2[labeled_mask2 == label] = 1

    # dbg.in_contact_debug(in_contact1, in_contact2)
    # print 'in contact 1', in_contact1
    # print 'in contact 2', in_contact2

    return in_contact1, in_contact2


def improved_watershed(binary_base, intensity, expected_separation=10):
    """
    Improved watershed method that takes in account minimum intensity as well as minimal size of
    separation between the elements

    :param binary_base: support for watershedding
    :param intensity: intensity value used to exclude  watershed points with too low of intensity
    :param expected_separation: expected minimal separation (in pixels) between watershed centers
    :return:
    """
    sel_elem = disk(2)

    # changed variable name for "labels"
    post_closing_labels = closing(binary_base, sel_elem)

    distance = ndi.distance_transform_edt(post_closing_labels)
    local_maxi = peak_local_max(distance,
                                indices=False,  # we want the image mask, not peak position
                                min_distance=expected_separation,  # about half of a bud with our size
                                threshold_abs=10,  # allows to clear the noise
                                labels=post_closing_labels)
    # we fuse the labels that are close together that escaped the min distance in local_maxi
    local_maxi = ndi.convolve(local_maxi, np.ones((5, 5)), mode='constant', cval=0.0)
    # finish the watershed
    expanded_maxi_markers = ndi.label(local_maxi, structure=np.ones((3, 3)))[0]
    segmented_cells_labels = watershed(-distance, expanded_maxi_markers, mask=post_closing_labels)

    unique_segmented_cells_labels = np.unique(segmented_cells_labels)
    unique_segmented_cells_labels = unique_segmented_cells_labels[1:]
    average_apply_mask_list = []
    for cell_label in unique_segmented_cells_labels:
        my_mask = segmented_cells_labels == cell_label
        apply_mask = segmented_cells_labels[my_mask]
        average_apply_mask = np.mean(intensity[my_mask])
        if average_apply_mask < 0.005:
            average_apply_mask = 0
            segmented_cells_labels[segmented_cells_labels == cell_label] = 0
        average_apply_mask_list.append(average_apply_mask)
    # x_labels = ['cell13', 'cell1', 'cell7', 'cell2', 'cell14', 'cell6', 'cell3', 'cell5', 'cell4', 'cell11', 'cell12', 'cell8', 'cell10', 'cell9']
    # dbg.improved_watershed_debug(segmented_cells_labels, intensity)
    # dbg.improved_watershed_plot_intensities(x_labels, average_apply_mask_list.sort())
    return segmented_cells_labels


def label_and_correct(binary_channel, value_channel, min_px_radius=3, min_intensity=0, mean_diff=10):
    """
    Labelling of a binary image, with constraints on minimal feature size, minimal intensity of area
     covered by a binary label or minimal mean difference from background

    :param binary_channel:
    :param value_channel: used to compute total intensity
    :param min_px_radius: minimal feature size
    :param min_intensity: minimal total intensity
    :param mean_diff: minimal (multiplicative) difference from the background
    :return:
    """
    labeled_field, object_no = ndi.label(binary_channel, structure=np.ones((3, 3)))
    background_mean = np.mean(value_channel[labeled_field == 0])

    for label in range(1, object_no+1):
        mask = labeled_field == label
        px_radius = np.sqrt(np.sum((mask).astype(np.int8)))
        total_intensity = np.sum(value_channel[mask])
        label_mean = np.mean(value_channel[labeled_field == label])
        if px_radius < min_px_radius or total_intensity < min_intensity or label_mean < mean_diff*background_mean:
            labeled_field[labeled_field == label] = 0
    # dbg.label_and_correct_debug(labeled_field)
    return labeled_field


def qualifying_gfp(max_sum_projection):
    """
    Creates a binary mask for qualifying gfp

    :param max_sum_projection:
    :return: binary mask
    """
    return max_sum_projection > 0


def label_based_aq(labels, field_of_interest):
    """
    Calculates average qualifying intensity of field of interest based on the labels mask

    :param labels:
    :param field_of_interest:
    :return: list of averages, pad of averages
    """
    average_list = []
    average_pad = np.zeros_like(labels).astype(np.float32)

    ########
    # TODO: REMOVE ME
    labels, _ = ndi.label(labels)

    #########

    for i in range(1, np.max(labels) + 1):

        current_mask = labels == i
        values_in_field = field_of_interest[current_mask]

        if len(values_in_field) == 0:
            continue

        _average = np.average(values_in_field)
        average_list.append(_average)
        average_pad[current_mask] = _average

    return np.array(average_list), average_pad


def average_qualifying_value_per_region(region_labels, image_2d, qualifying_mask):
    """
    Calculates average qualifying value per region of interest

    :param region_labels:
    :param image_2d:
    :param qualifying_mask:
    :return: np.array list of average values, 2d pad of average values
    """

    cells_average_gfp_list = []
    average_gfp_pad = np.zeros_like(region_labels).astype(np.float32)

    for i in range(1, np.max(region_labels) + 1):

        current_mask = region_labels == i
        current_cell_gfp = image_2d[np.logical_and(current_mask, qualifying_mask)]

        if len(current_cell_gfp) == 0:
            continue

        gfp_percentile = np.percentile(current_cell_gfp, 50)
        gfp_average = np.average(image_2d[np.logical_and(current_mask, image_2d > gfp_percentile)])
        cells_average_gfp_list.append(gfp_average)
        average_gfp_pad[current_mask] = gfp_average

    return np.array(cells_average_gfp_list), average_gfp_pad


def detect_upper_outliers(value_list):
    """
    Performs upper outlier detection based on the extreme value distribution intuition. Works best
    with over 15 data elements

    :param value_list:
    :return: positions of non-outliers, baseline curve of sorted averages, error margins
    """
    arg_sort = np.argsort(np.array(value_list))
    value_list = sorted(value_list)
    cell_no = range(0, len(value_list))
    # Non-trivial logic selecting the regression basis
    regression_base = min(len(value_list) - 3, 10)

    slope, intercept, _, _, _ = stats.linregress(np.array(cell_no)[1:regression_base],
                                                 np.array(value_list)[1:regression_base])

    error_margins = (np.max(np.array(value_list)[1:regression_base]) -
               np.min(np.array(value_list)[1:regression_base])) / 2
    error_margins *= 8

    predicted_average_gfp = intercept + slope * np.array(cell_no)

    non_outliers = arg_sort[np.array(cell_no)[np.array(predicted_average_gfp + error_margins) >
                                              np.array(value_list)]]
    return non_outliers, predicted_average_gfp, error_margins


def paint_mask(label_masks, labels_to_paint):
    """
    Paints a labeled mask based off a numpy list of values assigned to labels to paint.

    :param label_masks:
    :param labels_to_paint: 1d numpy array with values that need to be painted on the labels.
    :return:
    """
    #label mask is GFP upper outlier cells

    mask_to_paint = np.zeros_like(label_masks).astype(np.uint8)

    if labels_to_paint.tolist() != np.array([]):
        for idx in labels_to_paint.tolist():
            mask_to_paint[label_masks == idx + 1] = 1  # indexing starts from 1, not 0 for the labels
    return mask_to_paint


def mask_filter_2d(base, _filter):
    """
    Applies a filter a base mask in 2d

    :param base:
    :param _filter:
    :return:
    """
    ret_val = np.zeros_like(base)
    ret_val[_filter.astype(np.bool)] = base[_filter.astype(np.bool)]

    return ret_val


def clear_based_on_2d_mask(stack, mask):
    """
    Sets to 0 in 3d everything covered by the mask along the z axis

    :param stack: 3d image stack
    :param mask:
    :return:
    """
    return f_3d_stack_2d_filter(stack, np.logical_not(mask))


def binarize_3d(floats_volume, cutoff):
    """
    Performs a 3d binarization

    :param floats_volume:
    :param cutoff:
    :return:
    """
    binary_volume = np.zeros_like(floats_volume)
    binary_volume[floats_volume > cutoff] = 1
    return binary_volume.astype(np.bool)


def volume_mqvi(float_volume, binary_volume):
    """
    Calculates the median of the float volume after filtering it through the binary volume

    :param float_volume:
    :param binary_volume:
    :return:
    """
    m_q_v_i = np.median(float_volume[binary_volume])
    return m_q_v_i


def volume_aqvi(float_volume, binary_volume):
    """
    Calculates the average of the float volume after filtering it through the binary volume

    :param float_volume:
    :param binary_volume:
    :return:
    """
    a_q_v_i = np.mean(float_volume[binary_volume])
    return a_q_v_i


def otsu_tresholding(shape_base):
    """
    perofrms an otsu thresholding based of the shape_base

    :param shape_base:
    :return:
    """
    otsu = threshold_otsu(shape_base)
    ret_val = shape_base > otsu
    ret_val = ret_val.astype(np.bool)
    return ret_val


def binarize_2d(float_surface, cutoff_type='static', mcc_cutoff=None):
    """
    Performs a 2d binarization based on several possible methods

    :param float_surface:
    :param cutoff_type: ['otsu', 'local_otsu', 'static', 'log-otsu"]. Local Otsu is done with 5px mask
    :param mcc_cutoff: is cutoff_type is 'static', this will be the cutoff threshold
    :return: binary labels
    """
    if cutoff_type == 'otsu':
        mcc_cutoff = threshold_otsu(float_surface)

    elif cutoff_type == 'local otsu':
        selem = disk(5)
        mcc_cutoff = rank.otsu(float_surface, selem)

    elif cutoff_type == 'static':
        pass

    elif cutoff_type == 'log-otsu':
        mcc_cutoff = threshold_otsu(np.log(float_surface + np.min(float_surface[float_surface > 0])))

    else:
        raise PipeArgError('unknown cutoff type')

    binary_stack = np.zeros_like(float_surface).astype(np.bool)
    binary_stack[float_surface > mcc_cutoff] = 1

    return binary_stack


def agreeing_skeletons(float_surface, mito_labels):
    """
    Calculates agreeing skeletonization by both median and morphological skeletons

    :param float_surface: float volume on which we need to calculate the values
    :param mito_labels: labels that will be skeletonized
    :return:
    """
    topological_skeleton = skeletonize(mito_labels)

    medial_skeleton, distance = medial_axis(mito_labels, return_distance=True)

    # TODO: test without the active threshold surface
    active_threshold = np.mean(float_surface[mito_labels])
    transform_filter = np.zeros(mito_labels.shape, dtype=np.uint8)
    transform_filter[np.logical_and(medial_skeleton > 0, float_surface > active_threshold)] = 1
    # transform filter is basically medial_skeleton on a field above threshold (mean*5 - wow, that's a lot)
    medial_skeleton = transform_filter * distance

    median_skeleton_masked = np.ma.masked_array(medial_skeleton, medial_skeleton > 0)
    skeleton_convolve = ndi.convolve(median_skeleton_masked, np.ones((3, 3)),
                                     mode='constant', cval=0.0)
    divider_convolve = ndi.convolve(transform_filter, np.ones((3, 3)),
                                    mode='constant', cval=0.0)
    skeleton_convolve[divider_convolve > 0] = skeleton_convolve[divider_convolve > 0] / \
                                              divider_convolve[divider_convolve > 0]

    skeletons = np.zeros_like(medial_skeleton)
    skeletons[topological_skeleton] = skeleton_convolve[topological_skeleton]
    # dbg.skeleton_debug(float_surface, mito_labels, skeletons)
    return skeletons


def classify_fragmentation_for_mitochondria(label_mask, skeletons):
    """
    Performs mitochondria fragmentation based off the labels mask and skeletons mask

    :param label_mask:
    :param skeletons:
    :return:
    """
    # what if no mitochondria currently found?
    # what if we want to compare the surface of fragmented mitochondria v.s. non-fragmented ones?

    # well, one thing for sure, there is no way of escalating the skeleton/mito supression if they
    # are too small => we will need a filter on the label

    # maybe it actually is a good idea to get the mask manipulation for all areas in the skeleton

    # dbg.weight_sum_zero_debug(label_mask, skeletons)
    mask_items = np.unique(label_mask)
    mask_items = mask_items[mask_items > 0].tolist()

    radius_mask = np.zeros_like(label_mask).astype(np.float)
    support_mask = np.zeros_like(label_mask).astype(np.float)
    classification_mask = np.zeros_like(label_mask).astype(np.int)
    classification_roll = []
    weights = []

    for label in mask_items:
        px_radius = np.sqrt(np.sum((label_mask == label).astype(np.int)))
        support = np.sum((skeletons[label_mask == label] > 0).astype(np.int))

        if px_radius < 5 or support < 10:
            classification = 1  # fragment of a broken mitochondria
        else:
            classification = -1  # mitochondria is intact

        radius_mask += (label_mask == label).astype(np.float)*px_radius
        support_mask += (label_mask == label).astype(np.float)*support
        classification_mask += (label_mask == label).astype(np.int)*classification

        classification_roll.append(classification)
        weights.append(px_radius)

    classification_roll = np.array(classification_roll)
    final_classification = np.average(classification_roll, weights=weights)

    return final_classification, classification_mask, radius_mask, support_mask


def locally_normalize(channel, local_xy_pool=5, local_z_pool=2):
    """
    Performs a per- zslice local normalization of a 3d channel

    :param channel:
    :param local_xy_pool: size of the neighborhood to be considered for normalization
    :param local_z_pool: placeholder, currently unused
    :return: normalized 3d channel
    """
    selem = disk(local_xy_pool)

    new_slice_collector = []
    for xy_slice_index in range(0, channel.shape[0]):
        slice_median = median(channel[xy_slice_index, :, :], selem)
        new_slice = channel[xy_slice_index, :, :]*2**8 - slice_median  # TODO => resolve the int shen.

        new_slice[new_slice < 0] = 0
        new_slice[slice_median == 0] = 0

        # main_ax = plt.subplot(131)
        # plt.imshow(channel[xy_slice_index, :, :], interpolation='nearest')
        # plt.colorbar()
        #
        # plt.subplot(132, sharex=main_ax, sharey=main_ax)
        # plt.imshow(slice_median, interpolation='nearest')
        # plt.colorbar()
        #
        # plt.subplot(133, sharex=main_ax, sharey=main_ax)
        # plt.imshow(new_slice, interpolation='nearest')
        # plt.colorbar()
        #
        # plt.show()

        # we assume the image is 8 bits and has not been normalized before. In the future this might break.
        new_slice_collector.append(new_slice.astype(np.float)/2.**8)

    return np.array(new_slice_collector)


# Start of function merge
class Graph:
	'''Class for creating graphs for 3d image segmentation'''
	def __init__(self):
		# default dictionary to store graph
		self.graph = defaultdict(list)


	def addEdge(self, origin, destination, bidirectional = False, self_connect = True):
		'''Function to add an edge to graph, can be set to bidirectional if desired
		Manual entry of each element

		:param origin: [int] start node ID
		:param destination: [int] end node ID
		:param bidirectional: [bool] bool indicating whether the connection is bidirectional
		:param self_connect: [bool] indicate whether the origin node connects to itself.
		'''
		# Append edge to dictionary of for point
		self.graph[origin].append(destination)
		# Append origin node edge to itself
		if self_connect:
			self.graph[origin].append(origin)
		# Append node edge to itself
		self.graph[destination].append(destination)
		# Append reverse direction if bidirectional
		if bidirectional:
			self.graph[destination].append(origin)
		# Remove duplicates
		self.graph[origin] = list(set(self.graph[origin]))
		self.graph[destination] = list(set(self.graph[destination]))


	def rmEdge(self, origin, destination):
		'''Function tries to delete an edge in a graph, conditional on if it exists

		:param origin: [int] origin node number
		:param destination: [int] Destination node number
		'''

		if self.path_exists(origin, destination):
			origin_connections = len(self.graph[origin])
			dest_connections = len(self.graph[destination])
			self.graph[origin].remove(destination)
			if origin == destination:
				pass
			else:

				if origin_connections == 1 and dest_connections == 1:
					pass
				else:
					self.graph[destination].remove(origin)
		else:
			raise Exception("Path from {} to {} does not exist".format(origin, destination))


	def connections2graph(self, connection_table, connection_direction, *exist_list):
		'''Function creates a bidirectional graph given a 2d table of connections between points

		:param connection_table: [nd.array] numpy binary adjacency matrix
		:param connection_direction: [np.ndarray] numpy matrix of m x n bools
		:param exist_list: [list] list of whether elements within the axes of the adjacency matrix exist
		'''
		if not exist_list:
			exist_list = np.ones(len(connection_table))
		else:
			exist_list = exist_list[0]

		x_dim, y_dim = connection_table.shape
		exists = np.outer(exist_list, exist_list.T)
		connection_table = exists * connection_table
		# print connection_table
		for x in xrange(x_dim):
			for y in xrange(y_dim):
				if connection_table[x, y] == 1:
					self.addEdge(x, y, bidirectional = connection_direction[x, y])
				else:
					pass


	def BFS(self, s):
		'''Function to print a BFS(Breadth First Traversal) of graph

		:param s: [int] query node ID number
		'''
		connections = []
		# If element is not even in graph, there is no way to start from it
		if not s in self.graph:
			return connections
		# Mark all the vertices as not visited
		visited = [False]*(len(self.graph))
		dict_visted = dict(zip(self.graph.keys(), visited))
		# print dict_visted
		# Create a queue for BFS
		queue = []

		# Mark the source node as visited and enqueue it
		queue.append(s)
		dict_visted[s] = True
		# # print queue
		while queue:
		# 	# Dequeue a vertex from queue and print it
			s = queue.pop(0)
			# print s,
			connections.append(s)
			# Get all adjacent vertices of the dequeued
			# vertex s. If a adjacent has not been visited,
			# then mark it visited and enqueue it
			for i in self.graph[s]:
				if dict_visted[i] == False:
					queue.append(i)
					dict_visted[i] = True
		return connections


	def path_exists(self, start, end):
		'''Given a start point and an end point, determine whether if the two points are connected by any path.

		:param start: [int] node ID for starting node
		:param end: [int] node ID for ending node
		'''
		if not start in self.graph or not end in self.graph:
			return False
		else:
			if start == end:
				return True
			else:
				connections = self.BFS(start)
				if any(v == end for v in connections):
					return True
				else:
					return False


	def get_self(self):
		'''Statement used for getting graph contents for printing and debugging
		'''
		return self.graph



def get_3d_neighbor_coords(tuple_location, size):
	'''Gets neighbors directly adjacent to target voxel. 1U distance max. Does not include diagonally adjacent neighbors

	:param tuple_location: [tuple] query location
	:param size: [tuple] size dimensions of the original image listed in order of Z, X, Y, to get rid of any points that exceed the boundaries of the rectangular prism space
	:return: [list] of [tuple] list of tuples indicating neighbor locations
	'''
	neighbors = []
	z, x, y = tuple_location
	zdim, xdim, ydim = size

	top = (z + 1, x, y)
	bottom = (z - 1, x, y)
	front = (z, x + 1, y)
	back = (z, x - 1, y)
	left = (z, x, y - 1)
	right = (z, x, y + 1)

	neighbors = [top, bottom, front, back, left, right]
	neighbors = [pt for pt in neighbors if (pt[0] >= 0 and pt[1] >= 0 and pt[2] >= 0) and (pt[0] < zdim and pt[1] < xdim and pt[2] < ydim)]

	return neighbors


def get_3d_neighbor_coords_3U(tuple_location, size):
	'''Gets neighbors all around target voxel. sqrt(3)U distance max. Includes diagonally adjacent neighbors

	:param tuple_location: [tuple] query location
	:param size: [tuple] size dimensions of the original image listed in order of Z, X, Y, to get rid of any points that exceed the boundaries of the rectangular prism space
	:return: [list] of [tuple] list of tuples indicating neighbor locations
	'''
	neighbors = []
	z, x, y = tuple_location
	zdim, xdim, ydim = size

	top = (z + 1, x, y)
	bottom = (z - 1, x, y)
	front = (z, x + 1, y)
	back = (z, x - 1, y)
	left = (z, x, y - 1)
	right = (z, x, y + 1)

	corner1 = (z - 1, x - 1, y - 1)
	corner2 = (z - 1, x + 1, y - 1)
	corner3 = (z - 1, x + 1, y + 1)
	corner4 = (z - 1, x - 1, y + 1)
	corner5 = (z + 1, x - 1, y - 1)
	corner6 = (z + 1, x + 1, y - 1)
	corner7 = (z + 1, x + 1, y + 1)
	corner8 = (z + 1, x - 1, y + 1)

	edge1 = (z - 1, x, y - 1)
	edge2 = (z - 1, x + 1, y)
	edge3 = (z - 1, x, y + 1)
	edge4 = (z - 1, x - 1, y)
	edge5 = (z, x - 1, y - 1)
	edge6 = (z, x + 1, y - 1)
	edge7 = (z, x + 1, y + 1)
	edge8 = (z, x - 1, y + 1)
	edge9 = (z + 1, x, y -1)
	edge10 = (z + 1, x + 1, y)
	edge11 = (z + 1, x, y + 1)
	edge12 = (z + 1, x - 1, y)

	neighbors = [top, bottom, front, back, left, right,
					corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8,
					edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8, edge9, edge10, edge11, edge12]

	neighbors = [pt for pt in neighbors if (pt[0] >= 0 and pt[1] >= 0 and pt[2] >= 0) and (pt[0] < zdim and pt[1] < xdim and pt[2] < ydim)]

	return neighbors


def imglattice2graph(input_binary):
	'''Converts a 3d image into a graph for segmentation

	:param input_binary: [np.ndarray] complete binary image 3d
	:return item_id: [np.ndarray] indicies of all elements in the lattice for identification
	:return graph_map: [graph object] graph object indicating which voxels are connected to which voxels
	'''
	zdim, xdim, ydim = input_binary.shape
	# Instantiate graph
	graph_map = Graph()
	# Create an array of IDs
	item_id = np.array(range(0, zdim * xdim * ydim)).reshape(zdim, xdim, ydim)
	# Traverse input binary image
	# print "\tSlices Analyzed: ",
	for label in set(input_binary.flatten()):
		if label != 0:
			label_locations = [tuple(point) for point in np.argwhere(input_binary == label)]
			for location in label_locations:
				# Get Query ID Node #
				query_ID = item_id[location]
				# Get neighbors to Query
				neighbor_locations = get_3d_neighbor_coords(location, input_binary.shape)
				# For each neighbor
				for neighbor in neighbor_locations:
					# Get Neighbor ID
					neighbor_ID = item_id[neighbor]
					# If query exists and neighbor exists, branch query and neighbor.
					# If only Query exists, branch query to itself.
					if input_binary[neighbor]:
						graph_map.addEdge(origin = query_ID,
											destination = neighbor_ID,
											bidirectional = False,
											self_connect = True)
					else:
						graph_map.addEdge(origin = query_ID,
											destination = query_ID,
											bidirectional = False,
											self_connect = True)
		else:
			pass
	return item_id, graph_map


def layer_comparator(image3D):
	'''Uses lattice graph data to determine where the unique elements are and prune redundancies.

	:param image3D: [np.ndarray] original binary image 3d
	:return: [np.ndarray] segmented 3d image
	'''
	print "> Generating lattice"
	ID_map, graph = imglattice2graph(image3D)

	graph_dict = graph.get_self()
	# for key in sorted(graph_dict.iterkeys()):
	# 	print "%s: %s" % (key, graph_dict[key])
	network_element_list = []
	print "> Network size: ", len(graph_dict)
	# print graph_dict
	print "> Pruning Redundancies"
	for key in graph_dict.keys():
		try:
			network = sorted(graph.BFS(key))
			for connected_key in network:
				graph_dict.pop(connected_key, None)
			if network not in network_element_list:
				network_element_list.append(network)
		except:
			pass
	print "> Unique Paths + Background [1]: ", len(network_element_list)

	img_dimensions = ID_map.shape
	output = np.zeros_like(ID_map).flatten()

	last_used_label = 1
	print "> Labeling Network"
	for network in network_element_list:
		for element in network:
			output[element] = last_used_label
		last_used_label += 1
	return output.reshape(img_dimensions)


def euclid_dist_nD(p0, p1):
	'''Determines the euclidian distance between two n dimensional points

	:param p0: [tuple] point 0 tuple form
	:param p1: [tuple] point 1 tuple form
	:return: [float] distance
	'''
	return np.sum((p1 - p0) ** 2) ** 0.5


class Point_set(object):
	'''Class creates a set of points and a set of associated attributes with the point set.
	Intended to be used for triangles
	Points must be passed to the class in order.
	'''
	def __init__(self, point_list):
		self.point_list = np.array([[float(coordinate) for coordinate in point] for point in point_list])
		self.num_pts = len(self.point_list)


	def perimeter(self):
		'''returns the perimeter of the point set (assumes order in which the points were passed)

		:return: [float] perimeter value
		'''
		peri_distance = 0
		for pt_indx in xrange(self.num_pts):
			peri_distance += euclid_dist_nD(self.point_list[pt_indx],
											self.point_list[pt_indx - 1])
		return peri_distance


	def side_lengths(self):
		'''Determines the lengths of a each of the side lenths within the point set for the geometry they describe

		:return: [list] of side lengths point to point in the order they are listed.
		'''
		side_len = []
		for pt_indx in xrange(self.num_pts):
			side_len.append(euclid_dist_nD(self.point_list[pt_indx],
											self.point_list[pt_indx - 1]))
		return np.array(side_len)


	def heron_area(self):
		'''Intended to be used to determine the area of a triangle described by the point set (3 pts)
		DOES NOT WORK FOR ANY OTHER 2D GEOMETRIES

		:return: [float] area of prescribed triangle
		'''
		semi_peri = self.perimeter() / 2
		prod = semi_peri
		for side in self.side_lengths():
			prod *= semi_peri - side
		return np.sqrt(prod)


class Surface(object):
	'''class that creates a surface given a set of faces and verticies associated with each face.
	'''
	def __init__(self, triangle_collection):
		self.triangle_collection = triangle_collection
		self.num_triangles = len(triangle_collection)
		self.SA = self.get_SA()


	def get_SA(self):
		'''Determines the total surface area of the triangles in the collection of faces and verticies

		:return: [float] area of all triangles in a surface
		'''
		total = 0
		for triangle in self.triangle_collection:
			triangle_set = Point_set(triangle)
			total += triangle_set.heron_area()
		return total


	def get_stats(self):
		'''Returns the number of triangles in the surface and the total surface area

		:return: [int]
		'''
		return self.num_triangles, self.SA


def get_attributes(masked_image, x = 1.0, y = 1.0, stack_height = 1.0):
	'''Gets the attributes of a single binary element in 3d space.

	:param masked_image: [np.ndarray] binary 3d image
	:param x: [float] scaling factor for x (if calculation is desired in another unit than px)
	:param y: [float] scaling factor for y (if calculation is desired in another unit than px)
	:param stack_height: [float] scaling factor for z (if calculation is desired in another unit than px).
							This is also the distance between slices in a stack image
	:return volume, nTriangles, surfaceArea: [float], [int], [float] volume, number of triangles, and surface area of object
	'''
	masked_image[masked_image > 0] = 1
	volume = np.sum(masked_image) * stack_height

	masked_image = masked_image.astype(bool)
	# print "> Computing surface..."

	verts, faces, normals, values = measure.marching_cubes_lewiner(masked_image,
																	level = None,
																	spacing = (x, y, stack_height),
																	gradient_direction = 'descent',
																	step_size = 1,
																	allow_degenerate = True,
																	use_classic = False)
	triangle_collection = verts[faces]
	# print "> Computing attributes..."
	triangle_Surface = Surface(triangle_collection)
	nTriangles, surfaceArea = triangle_Surface.get_stats()
	return volume, nTriangles, surfaceArea


def reverse_cantor_pair(z):
	'''Decoder for determining which two numbers make the number pairing.

	:param z: [float] cantor number result
	:return x, y: [int], [int] number pairing that resulted in the cantor number result
	'''
	# Used for determining which cell and mitochondria ID correspond to a labeled mitochondria
	w = np.floor((np.sqrt((8 * z) + 1) - 1) / 2)
	t = (w ** 2 + w) / 2
	return int(w - z + t), int(z - t)



dtype2bits = {'uint8': 8,
              'uint16': 16,
              'uint32': 32}
