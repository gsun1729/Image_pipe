import os, errno
import re
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from scipy.misc import imsave
import skimage.io
import uuid
img_suffix = ".tif"


def get_img_filenames(root_directory, suffix = '.TIF'):
	'''
	Given a root directory, traverses all sub_directories and recovers any files with a given suffix.
	Assigns a UUID to each of the files found

	:param root_directory: [str] location to search
	:param suffix: [str] type of image suffix to look for
	:return: [list] a list of image files and their corresponding properties
	'''
	img_filelist = []
	for current_location, sub_directories, files in os.walk(root_directory):
		if files:
			for img_file in files:
				if (suffix.lower() in img_file.lower()) and '_thumb_' not in img_file:
					img_filename = img_file.replace(suffix, '')
					unique_ID = str(uuid.uuid4().hex)
					path_difference = os.path.relpath(current_location, root_directory)
					img_filelist.append((unique_ID,
										img_filename,
										img_file,
										path_difference,
										current_location,
										os.path.join(current_location, img_file)))
	return img_filelist


def get_just_filenames(root_directory, suffix = '.mat'):
	'''
	Given directory with datafiles, get just imagefilenames
	:param root_directory: [str] root search directory.  Note this should be the only directory, with no subdirectories for analysis to work
	:param suffix: [str] type of file to search for
	:return: [list] of filenames
	'''
	img_filelist = []
	for current_location, sub_directories, files in os.walk(root_directory):
		if files:
			for img_file in files:
				if (suffix.lower() in img_file.lower()) and '_thumb_' not in img_file:
					img_filelist.append(img_file)
	return img_filelist


def save_data(data, filename, write_directory):
	'''
	saves data to a filename in a write directory
	Saves a .mat file

	:param data: [np.ndarray] data to be saved
	:param filename: [str] file to be saved to
	:param write_directory: [str] directory file will be saved in
	'''
	save_dir = os.path.join(write_directory, filename)
	scipy.io.savemat(save_dir, mdict = {'data': data})
	# print filename, write_directory
	print "> Image Data '{}' saved to '{}'".format(filename, write_directory)


def save_figure(fig, name, write_directory):
	'''
	saves data to a filename in a write directory
	Saves a PNG file

	:param data: [np.ndarray] data to be saved
	:param filename: [str] file to be saved to
	:param write_directory: [str] directory file will be saved in
	'''
	imsave(os.path.join(write_directory,name), fig)
	print "> Image Figure '{}' saved to '{}'".format(name, write_directory)


def filepath2name(filepath):
	'''
	Turns a filename into just the string part

	Removes any OS identifiers

	:param filepath: [str] original filpath
	:return: [str] filepath concatenated
	'''
	if filepath[0] == ".":
		filepath = list(filepath)
		filepath[0] = ""
		filepath = "".join(filepath)
	filepath = filepath.replace("\\","_")
	filepath = filepath.replace(" ","-")
	return filepath


def mkdir_check(directory):
	'''
	Check to see if a directory exists, and if not, make the directory

	:param directory: [str] location for the directory to be made
	'''
	if not os.path.exists(directory):
		try:
			os.makedirs(directory)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise


def write_list_txt(location, filename, array):
	'''
	Given an array, write data to filename at location.

	:param location: [str] save directory for output file
	:param filename: [str] name of the output file
	:param array: [np.ndarray] data to be saved to file.
	'''
	writefile = open(os.path.join(location, filename), 'w')
	for row in array:
		for row_ele in xrange(len(row)):
			writefile.write(row[row_ele]+"\t")
		writefile.write("\n")
	writefile.close()


def read_txt_file(location):
	'''
	Reads the lookuptable generated from the main part of the algorithm, strips any new lines and tabs
	Used for reading LUT tables and cell mito UUID pairs

	:param location: [str] directory txt file is located in, .txt file
	:return: [list] of <lists> which includes data from the txt file.
	'''
	file_object = open(location, 'r')
	content = file_object.readlines()
	file_object.close()
	return [x.strip('\n').split('\t') for x in content]
