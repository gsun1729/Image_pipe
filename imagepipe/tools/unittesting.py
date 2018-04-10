import unittest
import numpy as np
import lib.pathfinder as pf
import lines.mito_counter as mcount
from skimage.data import binary_blobs
from lib.render import stack_viewer

class pathfinderTestCase(unittest.TestCase):
	"""Tests for pathfinder.py in function libraries"""
	def test_Graph(self):
		'''Test basic case of making a graph, adding edges, and removing edges'''
		self.assertRaises(pf.Graph())
		graph1 = pf.Graph()
		# Test creating a bidirectional graph
		for o_index in xrange(0,10):
			for d_index in xrange(3,10):
				self.assertRaises(graph1.addEdge(o_index,
													d_index,
													bidirectional = True,
													self_connect = True),
													msg = '{} cannot be connected to {}'.format(o_index, d_index))
		# Test removing edges from a bidirectional graph
		for o_index in xrange(2,4):
			for d_index in xrange(4,6):
				self.assertRaises(graph1.rmEdge(o_index, d_index),
									msg = '{} connection to {} cannot be removed'.format(o_index, d_index))
		# Test adding single edge and removing it
		self.assertRaises(graph1.addEdge(100, 120, bidirectional = True, self_connect = True))
		self.assertRaises(graph1.rmEdge(100, 100))
		self.assertRaises(graph1.addEdge(50, 150, bidirectional = False, self_connect = False))
		self.assertRaises(graph1.rmEdge(50,150))


	def test_GraphConnections(self):
		'''Test creating a graph given an incidence array and a bidirectional bool array'''
		paths = np.array([[1, 1, 1, 0, 1, 0, 0, 0],
						 [1, 1, 0, 1, 0, 1, 0, 0],
						 [1, 0, 1, 1, 0, 0, 1, 0],
						 [0, 1, 1, 1, 0, 0, 0, 1],
						 [1, 0, 0, 0, 1, 1, 1, 0],
						 [0, 1, 0, 0, 1, 1, 0, 1],
						 [0, 0, 1, 0, 1, 0, 1, 1],
						 [0, 0, 0, 1, 0, 1, 1, 1]])
		path_direction = np.zeros((8, 8), dtype = bool)
		graph2 = pf.Graph()
		self.assertRaises(graph2.connections2graph(paths, path_direction, np.array([0,0,1,1,0,1,1,1])))


	def test_BFS(self):
		graph = pf.Graph()
		graph.addEdge(1, 2, bidirectional = False, self_connect = False)
		graph.addEdge(1, 3, bidirectional = False, self_connect = False)
		graph.addEdge(1, 1, bidirectional = False, self_connect = False)
		graph.addEdge(1, 4, bidirectional = False, self_connect = False)
		graph.addEdge(1, 1, bidirectional = False, self_connect = False)
		graph.addEdge(5, 1, bidirectional = False, self_connect = False)
		graph.addEdge(5, 0, bidirectional = False, self_connect = False)
		graph.addEdge(4, 3, bidirectional = False, self_connect = False)
		graph.addEdge(4, 5, bidirectional = False, self_connect = False)
		graph.addEdge(4, 6, bidirectional = True, self_connect = False)
		graph.addEdge(0, 2, bidirectional = False, self_connect = False)
		self.assertEqual(graph.get_self()[0], [0, 2])
		self.assertEqual(graph.get_self()[1], [1, 2, 3, 4])
		self.assertEqual(graph.get_self()[2], [2])
		self.assertEqual(graph.get_self()[3], [3])
		self.assertEqual(graph.get_self()[4], [3, 4, 5, 6])
		self.assertEqual(graph.get_self()[5], [0, 1, 5])
		self.assertEqual(graph.get_self()[6], [4, 6])
		self.assertFalse(graph.path_exists(2,12))
		self.assertTrue(graph.path_exists(2,2))
		self.assertTrue(graph.path_exists(1,2))
		self.assertFalse(graph.path_exists(3,2))
		self.assertTrue(graph.path_exists(4,2))
		self.assertTrue(graph.path_exists(5,2))
		self.assertTrue(graph.path_exists(6,2))
		self.assertFalse(graph.path_exists(7,2))
		self.assertFalse(graph.path_exists(8,2))
		self.assertFalse(graph.path_exists(8,8))
		self.assertEqual(graph.BFS(0), [0, 2])
		self.assertEqual(sorted(graph.BFS(1)), [0, 1, 2, 3, 4, 5, 6])
		self.assertEqual(sorted(graph.BFS(2)), [2])
		self.assertEqual(sorted(graph.BFS(3)), [3])
		self.assertEqual(sorted(graph.BFS(4)), [0, 1, 2, 3, 4, 5, 6])
		self.assertEqual(sorted(graph.BFS(5)), [0, 1, 2, 3, 4, 5, 6])
		self.assertEqual(sorted(graph.BFS(6)), [0, 1, 2, 3, 4, 5, 6])

class mitoCounterTesting(unittest.TestCase):
	test_image = binary_blobs(length = 200,
								blob_size_fraction = 0.1,
								n_dim = 3,
								volume_fraction = 0.3,
								seed = 1)
	stack_viewer(test_image)

if __name__ == '__main__':
	unittest.main()
