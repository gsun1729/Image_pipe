from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import binary_blobs
from matplotlib.gridspec import GridSpec
import sys
from properties import *
'''Module includes functions for fast visualization of plots/image processing steps, 3d stack images for debugging
'''

def stack_viewer(image):
	'''
	Module to allow for scrolling through a 3d stack image modified from the following source:
	https://matplotlib.org/gallery/animation/image_slices_viewer.html

	:param image: [np.ndarray] 3d stack image for viewing
	'''
	try:
		z,x,y = image.shape
	except ValueError:
		print("Improper dimensions, non-stack Image")
		print(image.shape)
		sys.exit()

	class IndexTracker(object):
		def __init__(self, axes, image_stack):
			self.axes = axes
			axes.set_title('scroll to navigate images')

			self.image_stack = image_stack
			self.slices, rows, cols = image_stack.shape
			self.start_index = self.slices//2

			self.im = axes.imshow(self.image_stack[self.start_index,:, :])
			self.update()

		def onscroll(self, event):
			# print("%s %s" % (	event.button, event.step))
			if event.button == 'up':
				self.start_index = (self.start_index + 1) % self.slices
			else:
				self.start_index = (self.start_index - 1) % self.slices
			self.update()

		def update(self):
			self.im.set_data(self.image_stack[ self.start_index,:, :])
			axes.set_ylabel('slice %s' % self.start_index)
			self.im.axes.figure.canvas.draw()

	fig, axes = plt.subplots(1, 1)

	tracker = IndexTracker(axes, image)
	fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
	plt.show()


## 2d Stuff below here
def view_2d_img(img, save = False):
	'''Displays a single 2d image

	:param img: [np.ndarray] image for viewing
	:param save: [bool] save image or not
	'''
	fig = plt.figure()
	plt.imshow(img)
	if not save:
		plt.show()
	else:
		plt.savefig("asdf.png")
		# plt.close()


def make_ticklabels_invisible(fig):
	'''Helper function for montage_n_x, removes tick labels
	https://matplotlib.org/users/gridspec.html

	:param fig: [matplotlib.fig] figure to have tick labels removed
	'''
	for i, ax in enumerate(fig.axes):
		ax.text(0.5, 0.5, "ax%d" % (i + 1), va = "center", ha = "center")
		for tl in ax.get_xticklabels() + ax.get_yticklabels():
			tl.set_visible(False)


def montage_n_x(*tuple_img_line):
	'''Function takes a tuple of images to show a progression of images at each step in a processing
	pipeline.
	Multiple pipelines are displayed as individual rows, with each tuple submitted to the function
	representing a single pipeline.

	:param *tuple_img_line: [tuple] of images to be displayed in a row
	'''
	num_rows = len(tuple_img_line)
	num_cols = 0;
	for lines in tuple_img_line:
		if len(lines) > num_cols:
			num_cols = len(lines)
	# plt.figure()
	grid = GridSpec(num_rows, num_cols)
	for row in xrange(num_rows):
		for col in xrange(num_cols):
			try:
				plt.subplot(grid[row,col])
				properties(tuple_img_line[row][col])
				plt.imshow(tuple_img_line[row][col])
			except IndexError:
				print("Exceed index")
				break
		print("\n")
	make_ticklabels_invisible(plt.gcf())
	plt.show()


def plot_contour(points):
	'''
	Given a set of points in the format:
		[[1,2]
		 [1,2]
		 [3,2]
		 [4,3]]]
	plots the points in 2d space.
	:param points: [list] of [list]
	'''
	plt.plot(points[:, 0],  points[:, 1])
	plt.show()


def points2img(points):
	'''
	Given a set of points in the format:
		[[1,2]
		 [1,2]
		 [3,2]
		 [4,3]]]
	Creates an image of the points. (2d Numpy array)
	:param points: [list] of [list]
	:return: [np.ndarray]
	'''
	x_data = points[:, 0]
	y_data = points[:, 1]
	x_dim = int(np.ceil(np.amax(x_data) - np.amin(x_data)) + 1)
	y_dim = int(np.ceil(np.amax(y_data) - np.amin(y_data)) + 1)
	img = np.zeros((x_dim, y_dim))
	x_data = [int(np.floor(i)) - int(np.amin(x_data)) for i in x_data]
	y_data = [int(np.floor(j)) - int(np.amin(y_data)) for j in y_data]

	img[x_data, y_data] = 1
	return img


def render_contours(background, contour_list):
	'''Helper function for displaying contours detected in an image

	:param background: [np.ndarray] original image to show background
	:param contour_list: [list] list of contour locations and points
	'''
	fig, ax = plt.subplots()
	ax.imshow(background, interpolation = 'nearest', cmap = plt.cm.gray)
	for n, contour in enumerate(contour_list):
		ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
	plt.show()


def location(points):
	'''
	Given a set of points, determine what square in the image they lie in

	:param points: [list] of [list]
	'''
	x_data = points[:, 1]
	y_data = points[:, 0]
	top_left_x = int(np.ceil(np.amin(x_data)))
	top_left_y = int(np.ceil(np.amin(y_data)))
	bot_rite_x = int(np.ceil(np.amax(x_data)))
	bot_rite_y = int(np.ceil(np.amax(y_data)))

	return top_left_x, top_left_y, bot_rite_x, bot_rite_y


def img_px_histogram(data, nbins = None):
	'''
	Plots an image's pixel intensity distribution, takes in a 1d list

	:param data: [list]
	:param nbins: [int] number of bins for histogram
	'''
	n, bins, patches = plt.hist(data, nbins)
	plt.show()
	plt.xlabel('Pixel Intensity')
	plt.ylabel('Count')


def px_hist_stats_n0(image):
	'''Gets the values of each px within an image and calulates the mean px intensity and standard deviation
	Does not count px that have a value of 0

	:param image: [np.ndarray]
	:return: [float], [float]; mean and standard deviation
	'''
	data = image.flatten()
	f_data = [s for s in data if s != 0]
	return np.mean(f_data), np.std(f_data)


def px_stats(image):
	'''Gets the values of each px within an image and calulates the mean px intensity and standard deviation
	Does count px that have a value of 0

	:param image: [np.ndarray]
	:return: [float], [float]; mean and standard deviation
	'''
	data = image.flatten()
	return np.mean(data), np.std(data)


if __name__ == "__main__":
	'''Generates a 3d image of binary blobs and then runs the viewer.
	'''
	test_image = binary_blobs(length = 200,
								blob_size_fraction = 0.1,
								n_dim = 3,
								volume_fraction = 0.3,
								seed = 1)
	stack_viewer(test_image)
