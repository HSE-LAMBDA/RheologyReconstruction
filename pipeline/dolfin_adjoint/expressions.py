import numpy as np
from fenics import *
from fenics_adjoint import *

import math as m


class interpolant(UserExpression):

	def __init__(self, bbox, values, **kwargs):
		
		super().__init__()
		
		self.bbox = bbox
		self.values = values
		
		self.nx = int(values.shape[0])
		self.ny = int(values.shape[1])
		
		self.dx = (self.bbox[1][0] - self.bbox[0][0]) / float(self.nx)
		self.dy = (self.bbox[1][1] - self.bbox[0][1]) / float(self.ny)
		
	def eval(self, values, x):
		
		ix = int(x[0] / self.dx)
		iy = int(x[1] / self.dy)
		
		if ix == self.nx: ix -= 1
		if iy == self.ny: iy -= 1
		
		#print(2000. / self.dx)
		
		n1 = [ix - 1, iy - 1]
		n2 = [ix - 1, iy		]
		n3 = [ix - 1, iy + 1]
		n4 = [ix		, iy - 1]
		n5 = [ix		, iy]
		n6 = [ix		, iy + 1]
		n7 = [ix + 1, iy - 1]
		n8 = [ix + 1, iy		]
		n9 = [ix + 1, iy + 1]
		
		
		if (ix == 0): 
			n1[0] = 0
			n2[0] = 0
			n3[0] = 0
		if (iy == 0):
			n1[1] = 0
			n4[1] = 0
			n7[1] = 0
		if (ix >= self.nx - 1):
			n7[0] = self.nx - 1
			n8[0] = self.nx - 1
			n9[0] = self.nx - 1
		if (iy >= self.ny - 1):
			n3[1] = self.ny - 1
			n6[1] = self.ny - 1
			n9[1] = self.ny - 1
		
		neighbours = [n1, n2, n3, n4, n5, n6, n7, n8, n9]
		
		dists = np.array([
			m.sqrt(
				(x[0] - self.dx * n[0]) ** 2 +\
				(x[1] - self.dy * n[1]) ** 2
			) for n in neighbours
		])
		
		idx = np.argmin(dists)		 
		
		values[0] = self.values[neighbours[idx][0], neighbours[idx][1]]

	def value_shape(self): return ()


class ConstantLoad(UserExpression):
	"""
	A custom expression for describing the constant load function
	on the boundary of the region of integration
	"""
		
	def __init__(self, mesh, t, tc, p0, pulse_center, pulse_radius, **kwargs):
		"""
		:param mesh: 
			A triangular mesh. Required to look up for normal to a  given cell 
		:type mesh: dolfin.Mesh

		:param t: current time to evaluate the load function
		:type t: float

		:param tc: 
			cutoff time. Given (t > tc), the load becomes zero-valued
		:type tc: float

		:param p0: load quantity (in pascals)
		:type p0 : float

		"""
		super().__init__(**kwargs)

		self.mesh = mesh
	
		self.t  = t
		self.tc = tc
		self.p0 = p0
		self.pulse_center = pulse_center
		self.pulse_radius = pulse_radius
		
	def eval_cell(self, values, x, cell):
		
		normal = 0.0
		cell = Cell(self.mesh, cell.index)
		
		for f in facets(cell):
			if f.exterior(): normal = f.normal()
		
		if isinstance(normal, float): normal = (0., 1.) 
		factor = (-self.p0 * self.t/ self.tc) if (self.t <= self.tc) else  0.
		
		in_radius = np.sqrt(
			(x[0] - self.pulse_center[0]) ** 2 + (x[1] - self.pulse_center[1]) ** 2
		) <= self.pulse_radius

		values[0] = normal[0] * factor * in_radius
		values[1] = normal[1] * factor * in_radius

		
		
	def value_shape(self): return (2, )

