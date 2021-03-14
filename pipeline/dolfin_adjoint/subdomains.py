import numpy as np
from fenics import *
from fenics_adjoint import *


class boundary_x(SubDomain):
	"""
	Class, defining the boundary of rectangular area
	along the x axis
	"""
	def __init__(self, loc, **kwargs): 
		"""
		:param loc: x coord of the boundary
		:param tol: tolerance for proximity
		""" 
		super().__init__(**kwargs)
		self.loc = loc
		self.tol = 1e-14

	def inside(self, x, on_boundary): 
		return on_boundary and near(x[0], self.loc, self.tol)

class boundary_y(SubDomain):
	"""
	Class, defining the boundary of rectangular area
	along the y axis
	"""
	def __init__(self, loc, **kwargs):
		"""
		:param loc: y coord of the boundary
		:param tol: tolerance for proximity
		""" 
		super().__init__(**kwargs)
		self.loc = loc
		self.tol = 1e-14


	def inside(self, x, on_boundary): 
		return on_boundary and near(x[1], self.loc, self.tol)






	


