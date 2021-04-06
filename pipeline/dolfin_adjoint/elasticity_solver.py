import sys
sys.path.append('..')

import numpy as np
import yaml
import pprint

from functools import reduce
from yaml import Loader
from jsonschema import validate
from fenics import *
from .subdomains import *
from .expressions import *

from utils import isnotebook

__running_on_notebook__ = isnotebook()

if __running_on_notebook__: 
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm

from fenics_adjoint import *


__validation_schema__ = """
type: object
properties:
  mesh:
    type: object
    properties:
      bounding_box:
        type: object
        properties:
          p1:
            type: array
            items:
              type: number
            minItems: 2 
            maxItems: 2
          p2:
            type: array
            items:
              type: number
            minItems: 2 
            maxItems: 2
        required: [p1, p2]
      nx: 
        type: integer
        exclusiveMinimum: 1
      ny:
        type: integer
        exclusiveMinimum: 1
    required: [bounding_box, nx, ny]

  physics:
    type: object
    properties:
      lambda:
        type: object
        properties:
          scale_lambda: 
            type: number
            #minimum: '0.0'
          factor_lambda: 
            type: number
            #minimum: '0.0'
        required: [scale_lambda, factor_lambda]
      mu:
        type: object
        properties:
          scale_mu: 
            type: number
            #minimum: '0.0'
          factor_mu: 
            type: number
            #minimum: '0.0'
        required: [scale_mu, factor_mu]
      rho:
        type: object
        properties:
          scale_rho:
            type: number
            #minimum: '0.0'
          factor_rho:
            type: number
            #minimum: '0.0'
        required: [scale_rho, factor_rho]
    required: [lambda, mu, rho] 

  task:
    type: object
    properties:
      target_time: 
        type: number
        #minimum: '0.0'
      source:
        type: object
        properties:
          location: 
            type: string
            enum: [
              'top',
              'bottom',
              'left',
              'right'
            ]
          center:
            type: array
            items:
              type: number
            minItems: 2
            maxItems: 2 
          radius: 
            type: number
            #minimum: '0.0'
          magnitude: 
            type: number
            #minimum: '0.0'
          cutoff_time:
            type: number
        required: [location, center, radius, magnitude, cutoff_time] 
    required: [target_time, source]

  method:
    type: object
    properties:
      polynomial_type: 
        type: string
        enum: [
          'CG',
          'DG' 
        ]
      polynomial_order: 
        type: integer
      time_steps: 
        type: integer
      alphas:
        type: object
        properties:
          alpha_f: 
            type: number
            #minimum: '0.0'
          alpha_m: 
            type: number
            #minimum: '0.0'
        required: [alpha_f, alpha_m]
      LU_solver: 
        type: string
        enum: [
          'petsc'
        ]
    required: [polynomial_type, polynomial_order, time_steps, alphas]
required: [mesh, physics, task, method]
"""



class elasticity_solver():
  
    def __init__(
        self,
        values_lambda,
        values_mu,
        values_rho, 
        config_file='dolfin_adjoint/solver_config.yaml'):

        set_working_tape(Tape()) # TODO: check if works ok with multiple solvers

        with open(config_file, 'r') as f: config = yaml.load(f, Loader=Loader)
        validator = yaml.load(__validation_schema__, Loader=Loader)

        validate(config, validator)


        # mesh parameters
        self.bounding_box = [
            config['mesh']['bounding_box']['p1'],
            config['mesh']['bounding_box']['p2']
        ]

        self.nx = config['mesh']['nx']
        self.ny = config['mesh']['ny']
     
        # physics
        self.scale_lambda  = config['physics']['lambda']['scale_lambda']
        self.factor_lambda = config['physics']['lambda']['factor_lambda']
        self.scale_mu      = config['physics']['mu']['scale_mu']
        self.factor_mu     = config['physics']['mu']['factor_mu']
        self.scale_rho     = config['physics']['rho']['scale_rho']
        self.factor_rho    = config['physics']['rho']['factor_rho']

        # task
        self.T = config['task']['target_time']
        self.source_location  = config['task']['source']['location']
        self.source_center    = config['task']['source']['center']
        self.source_radius    = config['task']['source']['radius']
        self.source_magnitude = config['task']['source']['magnitude']
        self.cutoff_time      = config['task']['source']['cutoff_time']

        # method
        self.poly_type  = config['method']['polynomial_type']
        self.poly_order = config['method']['polynomial_order']
        self.time_steps = config['method']['time_steps']
        self.alpha_f    = config['method']['alphas']['alpha_f']
        self.alpha_m    = config['method']['alphas']['alpha_m']
        self.lu_solver  = config['method']['LU_solver']

        # post-init processing 

        # redefine physical parameters as constant functions
        self.scale_mu      = Constant(self.scale_mu)
        self.factor_mu     = Constant(self.factor_mu)
        self.scale_rho     = Constant(self.scale_rho)
        self.factor_rho    = Constant(self.factor_rho)
        self.scale_lambda  = Constant(self.scale_lambda)
        self.factor_lambda = Constant(self.factor_lambda)

        # add other useful constants

        # Rayleigh dampening constants, set to zero
        self.eta_m = Constant(0.)
        self.eta_k = Constant(0.)

        self.dt      = Constant(self.T / self.time_steps)
        self.alpha_f = Constant(self.alpha_f)
        self.alpha_m = Constant(self.alpha_m)
        self.gamma   = Constant(0.5 + self.alpha_f - self.alpha_m)
        self.beta    = Constant((self.gamma + 0.5) ** 2 / 4.)

        # rectangular mesh
        self.mesh = RectangleMesh(
            Point(self.bounding_box[0]),
            Point(self.bounding_box[1]), 
            self.nx,
            self.ny
        )
    
        self.V = VectorFunctionSpace(self.mesh, self.poly_type, self.poly_order) # base function space
        self.control_space = FunctionSpace(self.mesh, self.poly_type, self.poly_order) # control space

        # create boundary subdomains
        top   = boundary_y(self.bounding_box[1][1])
        bot   = boundary_y(self.bounding_box[0][1])
        left  = boundary_x(self.bounding_box[0][0])
        right = boundary_x(self.bounding_box[1][0])

        self.boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)

        top.mark(self.boundaries, 1)
        bot.mark(self.boundaries, 2)
        left.mark(self.boundaries, 3)
        right.mark(self.boundaries, 4)

        # set zero bounary conditions
        zero = Constant((0.0, 0.0))
        self.bcs = [
            DirichletBC(self.V, zero, left),
            DirichletBC(self.V, zero, right),
            DirichletBC(self.V, zero, bot)
        ]


        self.values_lambda = values_lambda
        self.values_mu     = values_mu
        self.values_rho    = values_rho


        # create distributions as interpolated expressions
        self.lmbda = interpolate(
            interpolant(self.bounding_box, self.values_lambda), self.control_space)
        self.mu    = interpolate(
            interpolant(self.bounding_box, self.values_mu), self.control_space)
        self.rho   = interpolate(
            interpolant(self.bounding_box, self.values_rho), self.control_space)


        self.dss = ds(subdomain_data=self.boundaries)
           
        if self.source_location == 'top':
            self.dss = self.dss(1)
        if self.source_location == 'bottom':
            self.dss = self.dss(2)
        if self.source_location == 'left':
            self.dss = self.dss(3)
        if self.source_location == 'right':
            self.dss = self.dss(4)              

    @staticmethod
    def local_project(v, V, u=None):
        dv = TrialFunction(V)
        v_ = TestFunction(V)

        a_proj = inner(dv, v_) * dx
        b_proj = inner(v,  v_) * dx

        solver = LocalSolver(a_proj, b_proj) 
        solver.factorize()

        if u is None:
            u = Function(V)
            solver.solve_local_rhs(u)
            return u
        else:
            solver.solve_local_rhs(u)
            return

    @staticmethod
    def avg(x_old, x_new, alpha): 
        return alpha * x_old + (1 - alpha) * x_new


    def sigma(self, r):
        """
        Stress tensor
        """
        return 2.0 * (self.scale_mu + self.factor_mu * self.mu) * sym(grad(r)) +\
               (self.scale_lambda + self.factor_lambda * self.lmbda) *\
               tr(sym(grad(r))) * Identity(len(r))

    def m(self, u, u_):
        """
        Mass matrix
        """
        return (self.scale_rho + self.factor_rho * self.rho)*inner(u, u_)*dx

    def k(self, u, u_):
        """
        Stiffness tensor
        """
        return inner(self.sigma(u), sym(grad(u_))) * dx

    def c(self, u, u_):
        """
        Rayleigh dampening   
        """
        return self.eta_m * self.m(u, u_) + self.eta_k * self.k(u, u_)

    @staticmethod
    def Wext(u_, p, ds):
        """
        Surface forces
        """
        return dot(u_, p) * ds

    def _update_a(self, u, u_old, v_old, a_old, ufl=True):
        if ufl:
            dt_   = self.dt
            beta_ = self.beta 
            return (u - u_old - dt_ * v_old) / beta_/ dt_ ** 2 -\
                   (1 - 2 * beta_)/2/beta_ * a_old 
        else:
            dt_   = float(self.dt)
            beta_ = float(self.beta)
        return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

    def _update_v(self, a, u_old, v_old, a_old, ufl=True):
        if ufl:
            dt_    = self.dt
            gamma_ = self.gamma
            return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

        else:
            dt_ = float(self.dt)
            gamma_ = float(self.gamma)
        return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

    def _update_fields(self, u, u_old, v_old, a_old):
        """
        Update fields at the end of each time step.
        """
        u_vec, u0_vec  = u.vector(), u_old.vector()
        v0_vec, a0_vec = v_old.vector(), a_old.vector()

        a_vec = self._update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
        v_vec = self._update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

        v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
        u_old.vector()[:] = u.vector()



    def _project_grads(self, grads):

        p1 = self.bounding_box[0]
        p2 = self.bounding_box[1]

        sl = self.values_lambda.shape
        dx_l = (p2[0] - p1[0]) / float(sl[0])
        dy_l = (p2[1] - p1[1]) / float(sl[1])

        lmbda_xx, lmbda_yy = np.meshgrid(
            np.linspace(p1[0] + dx_l / 2. , p2[0] - dx_l / 2., sl[0]),
            np.linspace(p1[1] + dy_l / 2. , p2[1] - dy_l / 2., sl[1]),
        )

        arr_lambda = [grads[0](x, y) for x, y in zip(lmbda_xx.flatten(), lmbda_yy.flatten())]
        arr_lambda = np.array(arr_lambda).reshape(*sl)

        sm = self.values_mu.shape
        dx_m = (p2[0] - p1[0]) / float(sm[0])
        dy_m = (p2[1] - p1[1]) / float(sm[1])

        mu_xx, mu_yy = np.meshgrid(
            np.linspace(p1[0] + dx_m / 2. , p2[0] - dx_m / 2., sl[0]),
            np.linspace(p1[1] + dy_m / 2. , p2[1] - dy_m / 2., sl[1]),
        )

        arr_mu = [grads[1](x, y) for x, y in zip(mu_xx.flatten(), mu_yy.flatten())]
        arr_mu = np.array(arr_mu).reshape(*sm)

        sr = self.values_rho.shape
        dx_r = (p2[0] - p1[0]) / float(sr[0])
        dy_r = (p2[1] - p1[1]) / float(sr[1])

        rho_xx, rho_yy = np.meshgrid(
            np.linspace(p1[0] + dx_r / 2. , p2[0] - dx_r / 2., sl[0]),
            np.linspace(p1[1] + dy_r / 2. , p2[1] - dy_r / 2., sl[1]),
        )

        arr_rho = [grads[2](x, y) for x, y in zip(rho_xx.flatten(), rho_yy.flatten())]
        arr_rho = np.array(arr_rho).reshape(*sr)

        return arr_lambda, arr_mu, arr_rho


    def forward(
        self, 
        save_callback=None
    ):

        # init constant load
        p = ConstantLoad(
            self.mesh, 0., self.cutoff_time, self.source_magnitude,
            self.source_center, self.source_radius
        )

       
        du = TrialFunction(self.V)
        u_ = TestFunction(self.V)
        u  = Function(self.V, name="Displacement")

        u_old = Function(self.V)
        v_old = Function(self.V)
        a_old = Function(self.V)

        a_new = self._update_a(du, u_old, v_old, a_old, ufl=True)
        v_new = self._update_v(a_new, u_old, v_old, a_old, ufl=True)
    
        res = self.m(self.avg(a_old, a_new, self.alpha_m), u_) +\
              self.c(self.avg(v_old, v_new, self.alpha_f), u_) +\
              self.k(self.avg(u_old, du, self.alpha_f), u_) -\
              self.Wext(u_, p, self.dss)
    
        a_form = lhs(res)
        L_form = rhs(res)

        # assemble matrix
        K, res = assemble_system(a_form, L_form, self.bcs)
        solver = LUSolver(K, self.lu_solver)

        time = np.linspace(0., self.T, self.time_steps)

        for (i, dt) in enumerate(np.diff(time)):
            t = time[i + 1]
            p.t = t - float(self.alpha_f * self.dt)

            res = assemble(L_form)
            for bc in self.bcs: bc.apply(res)

            solver.solve(K, u.vector(), res)
            self._update_fields(u, u_old, v_old, a_old)
            if save_callback is not None: save_callback(i, t, u_old, v_old, a_old)


class dolfin_adjoint_solver(elasticity_solver):
    """
    A class to compute gradient of an error functional
    algorithmically via dolfin-adjoint framework
    """

    def __init__(
        self,
        values_lambda,
        values_mu,
        values_rho, 
        detector_coords, 
        config_file='dolfin_adjoint/solver_config.yaml'
    ):

        super().__init__(values_lambda, values_mu, values_rho, config_file=config_file)
        self.detector_coords = detector_coords

        self.a1 = Constant(1. / self.beta / self.dt ** 2)
        self.a2 = Constant(-1. * self.dt / self.beta / self.dt ** 2)
        self.a3 = Constant(-(1 - 2 * self.beta) / 2. / self.beta)
        self.v1 = Constant(1.)
        self.v2 = Constant(self.dt * (1 - self.gamma))
        self.v3 = Constant(self.dt * self.gamma)


    # use costy functions
    # to preserve gradients
    # TODO: function stack grows very fast, and it looks like
    #       it is the reason of memory leakage. Is there a way to clean
    #       the stack explicitly?

    def _update_a(self, u, u_old, v_old, a_old, ufl=True):
        if ufl:
            dt_   = self.dt
            beta_ = self.beta 
            return (u - u_old - dt_ * v_old) / beta_/ dt_ ** 2 -\
                   (1 - 2 * beta_)/2/beta_ * a_old 

        return FunctionAXPY([
            (self.a1, u),
            (-self.a1, u_old),
            (self.a2, old),
            (self.a3, a_old)
        ], annotate=True)


    def _update_v(self, a, u_old, v_old, a_old, ufl=True):
        if ufl:
            dt_    = self.dt
            gamma_ = self.gamma
            return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

        return FunctionAXPY([
            (self.v1, v_old),
            (self.v2, a_old),
            (self.v3, a)
        ], annotate=True)

    
    def _update_fields(self, u, u_old, v_old, a_old):
        a = Function(self.V)
        a.assign(
            self.a1 * u -\
            self.a1 * u_old +\
            self.a2 * v_old +\
            self.a3 * a_old,
            annotate=True
        )

        v = Function(self.V)
        v.assign(
            self.v1 * v_old +\
            self.v2 * a_old +\
            self.v3 * a,
            annotate=True
        )

        u_old.assign(u, annotate=True) 
        v_old.assign(v, annotate=True)
        a_old.assign(a, annotate=True)


    def _forward(
        self,
        seismogram,
        save_callback=None
    ):
    
        assert(seismogram.shape[1] == self.time_steps)
        assert(seismogram.shape[2] == len(self.detector_coords))

        # init constant load
        p = ConstantLoad(
            self.mesh, 0., self.cutoff_time, self.source_magnitude,
            self.source_center, self.source_radius
        )
 
        # assemble bilinear and linear ufl forms
        du = TrialFunction(self.V)
        u_ = TestFunction(self.V)
        u  = Function(self.V, name="Displacement")

        u_old = Function(self.V)
        v_old = Function(self.V)
        a_old = Function(self.V)

        a_new = self._update_a(du, u_old, v_old, a_old, ufl=True)
        v_new = self._update_v(a_new, u_old, v_old, a_old, ufl=True)
    
        res = self.m(self.avg(a_old, a_new, self.alpha_m), u_) +\
              self.c(self.avg(v_old, v_new, self.alpha_f), u_) +\
              self.k(self.avg(u_old, du, self.alpha_f), u_) -\
              self.Wext(u_, p, self.dss)
    
        a_form = lhs(res)
        L_form = rhs(res)

        # assemble matrix
        K, res = assemble_system(a_form, L_form, self.bcs)
        solver = LUSolver(K, self.lu_solver)

        time = np.linspace(0., self.T, self.time_steps)

        # initialize error functional
        j = 0.

        for (i, dt) in enumerate(
            tqdm(np.diff(time), 
                 desc='integrating the state problem',
                 miniters=20
        )):

            t = time[i + 1]
            p.t = t - float(self.alpha_f * self.dt)

            res = assemble(L_form)
            for bc in self.bcs: bc.apply(res)

            solver.solve(K, u.vector(), res)
            self._update_fields(u, u_old, v_old, a_old)
            if save_callback is not None: save_callback(i, t, u_old, v_old, a_old)
   
            preds  = [u_old(c) for c in self.detector_coords]
            ground = seismogram[:, i+1].T
    
            for a, b in zip(preds, ground):
                j += 0.5 * ((float(b[0]) - a[0]) ** 2 + (float(b[1]) - a[1]) ** 2)


        controls = [Control(self.lmbda), Control(self.mu), Control(self.rho)]
        rf       = ReducedFunctional(j, controls)

        return float(j), rf

    def backward(
        self,
        seismogram,
        save_callback=None
    ):

        loss, reduced_functional = self._forward(seismogram)
        grads = reduced_functional.derivative()
        grad_lambda, grad_mu, grad_rho = self._project_grads(grads)
        
        return loss, (
            float(self.factor_lambda) * grad_lambda,
            float(self.factor_mu) * grad_mu,
            float(self.factor_rho) * grad_rho
        )



class adjoint_equation_solver(elasticity_solver):
    """
    A class to compute gradient of an error functional
    via solving the adjoint problem numerically
    Works significantly faster than dolfin_adjoint_solver,
    TODO: find out whether the gradient is precise enough to use it all the times
          instead of dolfin_adjoint_solver

    """
    def __init__(
        self,
        values_lambda,
        values_mu,
        values_rho, 
        detector_coords, 
        config_file='dolfin_adjoint/solver_config.yaml'
    ):

        super().__init__(values_lambda, values_mu, values_rho, config_file=config_file)
        self.detector_coords = detector_coords
        # outer normal to top boundary
        self.ny = Constant((0., 1.))

    def _forward(
        self,
        seismogram,
        save_callback=None,
    ):
    
        assert(seismogram.shape[1] == self.time_steps)
        assert(seismogram.shape[2] == len(self.detector_coords))

        # init constant load
        p = ConstantLoad(
            self.mesh, 0., self.cutoff_time, self.source_magnitude,
            self.source_center, self.source_radius
        )

        # build ufl bilinear and linear forms

        du = TrialFunction(self.V)
        u_ = TestFunction(self.V)
        u  = Function(self.V, name="Displacement")

        u_old = Function(self.V)
        v_old = Function(self.V)
        a_old = Function(self.V)

        a_new = self._update_a(du, u_old, v_old, a_old, ufl=True)
        v_new = self._update_v(a_new, u_old, v_old, a_old, ufl=True)
    
        res = self.m(self.avg(a_old, a_new, self.alpha_m), u_) +\
              self.c(self.avg(v_old, v_new, self.alpha_f), u_) +\
              self.k(self.avg(u_old, du, self.alpha_f), u_) -\
              self.Wext(u_, p, self.dss)
    
        a_form = lhs(res)
        L_form = rhs(res)

        # assemble matrix
        K, res = assemble_system(a_form, L_form, self.bcs)
        solver = LUSolver(K, self.lu_solver)

        time = np.linspace(0., self.T, self.time_steps)

        # initialize error functional
        j = 0.

        # buffer for adjoint sources history
        adj_source = np.zeros((seismogram.shape[1], len(self.detector_coords), 2))


        # values of displacement at detector locations
        preds  = np.array([u_old(c) for c in self.detector_coords])
        ground = seismogram[:, 0].T
        
        # adj sources are driven by misfit functional, so we save difference
        # between preds and ground truth
        adj_source[0] = np.array(preds) - ground


        # save displacement history while running
        disp_history  = [u_old.copy(deepcopy=True)]
        
        
        # run integration
        for (i, dt) in enumerate(
            tqdm(np.diff(time), 
                 desc='integrating the state problem',
                 miniters=20
        )):

            t = time[i + 1]
            p.t = t - float(self.alpha_f * self.dt)

            res = assemble(L_form)
            for bc in self.bcs: bc.apply(res)

            solver.solve(K, u.vector(), res)
            self._update_fields(u, u_old, v_old, a_old)
            if save_callback is not None: save_callback(i, t, u_old, v_old, a_old)

            preds  = np.array([u_old(c) for c in self.detector_coords])
            ground = seismogram[:, i+1].T
            adj_source[i+1] = preds - ground
            
            disp_history.append(u_old.copy(deepcopy=True))
    
            # add to misfit functional
            for a, b in zip(preds, ground): j += 0.5 * np.sum(np.power(b - a, 2))

        return j / float(self.time_steps), adj_source, disp_history


    def _adjoint_forward(self, adj_source, save_callback=None):

        # invert timestep
        self.dt = Constant(-self.dt)

        # build adjoint source as sum of delta functions
        p = AdjointLoad(self.mesh, self.T, self.T, adj_source, self.detector_coords)
 
        a1 = Constant(1. / self.beta / self.dt ** 2)
        a2 = Constant(-1. * self.dt / self.beta / self.dt ** 2)
        a3 = Constant(-(1 - 2 * self.beta) / 2. / self.beta)
        v1 = Constant(1.)
        v2 = Constant(self.dt * (1 - self.gamma))
        v3 = Constant(self.dt * self.gamma)

        # build ufl bilinear and linear forms

        du = TrialFunction(self.V)
        u_ = TestFunction(self.V)
        u  = Function(self.V, name="Displacement")

        u_old = Function(self.V)
        v_old = Function(self.V)
        a_old = Function(self.V)

        a_new = self._update_a(du, u_old, v_old, a_old, ufl=True)
        v_new = self._update_v(a_new, u_old, v_old, a_old, ufl=True)
    
        # change dampening sign
        # and add adj. boundary condition
        res = self.m(self.avg(a_old, a_new, self.alpha_m), u_) -\
              self.c(self.avg(v_old, v_new, self.alpha_f), u_) +\
              self.k(self.avg(u_old, du, self.alpha_f), u_) +\
              dot(self.sigma(self.avg(u_old, du, self.alpha_f)) * self.ny - p, u_) * self.dss
    
        a_form = lhs(res)
        L_form = rhs(res)

        # assemble matrix
        K, res = assemble_system(a_form, L_form, self.bcs)
        solver = LUSolver(K, self.lu_solver)

        # save adjoint history while running
        adj_history = [u_old.copy(deepcopy=True)]

        time = np.linspace(self.T, 0., self.time_steps)

        # integrate in reverse time
        for (i, dt) in enumerate(
            tqdm(np.diff(time), 
                 desc='integrating adjoint problem in reverse time', 
                 miniters=20
            )):
            
            t = time[i + 1]
            p.t = t - float(self.alpha_f * self.dt)

            res = assemble(L_form)
            for bc in self.bcs: bc.apply(res)

            solver.solve(K, u.vector(), res)
            self._update_fields(u, u_old, v_old, a_old)
            
            adj_history.append(u_old.copy(deepcopy=True))
            if save_callback is not None: save_callback(i, t, u_old, v_old, a_old)


        self.dt = Constant(-self.dt)
        adj_history.reverse()

        return adj_history


    def backward(
        self,
        seismogram,
        save_callback=None
    ):

        loss, adj_sources, disp_history = self._forward(seismogram)
        adjoint_history = self._adjoint_forward(adj_sources, save_callback=save_callback)
    

        # compute and project gradients wrt lambda, mu, rho

        grad_lambda = self.local_project(
            reduce(
                lambda x, y: x + y,
                [
                    -div(t) * div(a) * self.dt
                    for t, a in zip(disp_history, adjoint_history) 
                ]
            ),
            self.control_space
        )

        grad_mu = self.local_project(
            reduce(
                lambda x, y: x + y,
                [
                    -inner(grad(a), grad(t) + grad(t).T) * self.dt
                    for t, a in zip(disp_history, adjoint_history)
                ]
            ),
            self.control_space
        )

        grad_rho = self.local_project(
            reduce(
                lambda x, y: x + y,
                [
                    -inner(t, div(self.sigma(a))) * self.dt 
                    for t, a in zip(disp_history, adjoint_history)
                ] 
            ),
            self.control_space
        )

        grad_lambda, grad_mu, grad_rho = self._project_grads([grad_lambda, grad_mu,  grad_rho])

        return loss, (
            float(self.factor_lambda) * grad_lambda,
            float(self.factor_mu) * grad_mu,
            float(self.factor_rho) * grad_rho
        )