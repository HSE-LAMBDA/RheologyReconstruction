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

		if x[0] < self.bbox[0][0]:
			
			if x[1] > self.bbox[1][1]:
				values[0] = self.values[0, -1]
			elif x[1] < self.bbox[0][1]:
				values[0] = self.values[0, 0]
			else:
				iy = int(x[1] / self.dy)
				if iy == self.ny: iy -= 1
				values[0] = self.values[0, iy]

			return

		if x[0] > self.bbox[1][0]:
			
			if x[1] > self.bbox[1][1]:
				values[0] = self.values[-1, -1]
			elif x[1] < self.bbox[0][1]:
				values[0] = self.values[-1, 0]
			else:
				iy = int(x[1] / self.dy)
				if iy == self.ny: iy -= 1
				values[0] = self.values[-1, iy]

			return

		if x[1] < self.bbox[0][1]:

			if x[0] > self.bbox[1][0]:
				values[0] = self.values[-1, 0]
			elif x[0] < self.bbox[0][0]:
				values[0] = self.values[0, 0]
			else:
				ix = int(x[0] / self.dy)
				if ix == self.nx: ix -= 1
				values[0] = self.values[ix, 0]

			return

		if x[1] > self.bbox[1][1]:

			if x[0] > self.bbox[1][0]:
				values[0] = self.values[-1, -1]
			elif x[0] < self.bbox[0][0]:
				values[0] = self.values[0, -1]
			else:
				ix = int(x[0] / self.dy)
				if ix == self.nx: ix -= 1
				values[0] = self.values[ix, 0]

			return
		
		
		ix = int(x[0] / self.dx)
		iy = int(x[1] / self.dy)
		
		if ix == self.nx: ix -= 1
		if iy == self.ny: iy -= 1
		
		n1 = [ix - 1, iy - 1]
		n2 = [ix - 1, iy	]
		n3 = [ix - 1, iy + 1]
		n4 = [ix	, iy - 1]
		n5 = [ix	, iy	]
		n6 = [ix	, iy + 1]
		n7 = [ix + 1, iy - 1]
		n8 = [ix + 1, iy	]
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

		return

	def value_shape(self): return ()


class CustomLoad(UserExpression):
	"""
	A custom expression for describing the load function
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

	def value_shape(self): return (2, )
		
	def eval_cell(self, values, x, cell): 
		raise NotImplementedError


class ConstantLoad(CustomLoad):

	def eval_cell(self, values, x, cell):
		
		normal = 0.0
		cell = Cell(self.mesh, cell.index)
		
		for f in facets(cell):
			if f.exterior(): normal = f.normal()
		
		if isinstance(normal, float): normal = (0., 1.) 
		factor = -self.p0 if (self.t <= self.tc) else  0.
		
		in_radius = np.sqrt(
			(x[0] - self.pulse_center[0]) ** 2 + (x[1] - self.pulse_center[1]) ** 2
		) <= self.pulse_radius

		values[0] = normal[0] * factor * in_radius
		values[1] = normal[1] * factor * in_radius

		
class SineLoad(CustomLoad):
		
	def __init__(self, mesh, t, tc, p0, pulse_center, pulse_radius, period, **kwargs):
		
		super().__init__(
			mesh, t, tc, p0, pulse_center, pulse_radius, **kwargs
		)

		self.period = period
		
	
	def eval_cell(self, values, x, cell):
		
		normal = 0.0
		cell = Cell(self.mesh, cell.index)
		
		for f in facets(cell):
			if f.exterior(): normal = f.normal()
		
		if isinstance(normal, float): normal = (0., 1.) 

		factor = -self.p0 * np.cos(2 * np.pi * self.t / self.period) if (self.t <= self.tc) else  0.
		
		in_radius = np.sqrt(
			(x[0] - self.pulse_center[0]) ** 2 + (x[1] - self.pulse_center[1]) ** 2
		) <= self.pulse_radius

		values[0] = normal[0] * factor * in_radius
		values[1] = normal[1] * factor * in_radius


class GaussianLoad(CustomLoad):
		
	def __init__(
		self, mesh, t, tc, 
		p0, pulse_center, pulse_radius,
		period, mu, sigma, 
		**kwargs
	):
		
		super().__init__(mesh, t, tc, p0, pulse_center, pulse_radius, **kwargs)

		self.period = period
		self.mu     = mu
		self.sigma  = sigma
		
	def eval_cell(self, values, x, cell):
		
		normal = 0.0
		cell = Cell(self.mesh, cell.index)
		
		for f in facets(cell):
			if f.exterior(): normal = f.normal()
		
		if isinstance(normal, float): normal = (0., 1.) 

		factor_sine  = np.cos(2 * np.pi * self.t / self.period)
		factor_gauss = 1 / m.sqrt(np.pi * 2 * self.sigma) *\
					   np.exp(- ((self.t - self.mu) ** 2) / 2. / (self.sigma ** 2)) 
		
		factor = -self.p0 * factor_sine * factor_gauss if (self.t <= self.tc) else  0.
		
		in_radius = np.sqrt(
			(x[0] - self.pulse_center[0]) ** 2 + (x[1] - self.pulse_center[1]) ** 2
		) <= self.pulse_radius

		values[0] = normal[0] * factor * in_radius
		values[1] = normal[1] * factor * in_radius


class AdjointLoad(UserExpression):
	"""
	A custom expression to implement adjoint load,
	which emerges in derivation of Lagrange function stationarity

	"""

	@staticmethod
	def delta(x, p, scale, magnitude):
		r = (x[0] - p[0]) ** 2 + (x[1] - p[1]) ** 2
		factor = np.sqrt(1 / np.pi / scale) * np.exp(-r / 2. / scale ** 2)

		return magnitude * factor

	def __init__(self, mesh, t, integration_time, magnitudes, detector_coords, **kwargs):

		super().__init__(**kwargs)

		self.mesh = mesh
		self.t    = t
		self.integration_time = integration_time
		self.detector_coords  = detector_coords

		self.magnitudes = magnitudes

		self.dt  = self.integration_time / float(len(magnitudes))

		#stretch factor should be finetuned
		#SUGGESTION: make sanity check for regular problem on the same mesh
		
		self.a   = 1.

	def eval(self, values, x): # cell):

		i = min(len(self.magnitudes) - 1, m.floor(self.t / self.dt))
		j = min(len(self.magnitudes) - 1, m.ceil(self.t / self.dt))

		alpha = (self.t - self.dt * float(i)) / self.dt

		total_magnitude = np.zeros(2)

		for k, p in enumerate(self.detector_coords):
			total_magnitude += self.delta(
				x, p, self.a, self.magnitudes[i, k]*alpha + self.magnitudes[j, k]*(1-alpha))

		values[0] = total_magnitude[0]
		values[1] = total_magnitude[1]


	def value_shape(self): return (2, )