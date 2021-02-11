import os
import argparse

from fenics import *
from fenics_adjoint import *


set_log_level(0)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print '%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result
    return timed


class lambda_(UserExpression):
    
    def __init__(self, materials, lambda_0, lambda_1, **kwargs):
        
        super().__init__(**kwargs)
        
        self.materials = materials
        self.lambda_0  = lambda_0
        self.lambda_1  = lambda_1
        
    def eval_cell(self, values, x, cell):
        
        if self.materials[cell.index] == 0:
            values[0] = self.lambda_0
        else:
            values[0] = self.lambda_1
            
        return values
            
class mu_(UserExpression):
    
    def __init__(self, materials, mu_0, mu_1, **kwargs):
        
        super().__init__(**kwargs)
        
        self.materials = materials
        self.mu_0  = mu_0
        self.mu_1  = mu_1
        
    def eval_cell(self, values, x, cell):
        
        if self.materials[cell.index] == 0:
            values[0] = self.mu_0
        else:
            values[0] = self.mu_1

            
class rho_(UserExpression):
    
    def __init__(self, materials, rho_0, rho_1, **kwargs):
        
        super().__init__(**kwargs)
        
        self.materials = materials
        self.rho_0  = rho_0
        self.rho_1  = rho_1
        
    def eval_cell(self, values, x, cell):
        
        if self.materials[cell.index] == 0:
            values[0] = self.rho_0
        else:
            values[0] = self.rho_1


def main(mesh_name: str, mesh_tags: str):
	"""
	:args:
		mesh_name - name of the mesh file in current working directory
		mesh_tags - name of the mesh tags file in current working directory.
					Tags specify the division of the mesh on subdomains
	"""


	mpi_comm = df.mpi_comm_world()
	my_rank = df.MPI.rank(mpi_comm)

	if my_rank==0:
		mesh = df.Mesh()
   	else:
   	    mesh= None

	partition  = MeshFunction('int', mesh, 'test_mesh_small_cell_tags.xml')
	subdomains = MeshFunction('size_t', mesh, mesh.topology().dim())
	
	for c in cells(mesh):
		if partition[c] == -6: subdomains[c] = 0
		if partition[c] == -7: subdomains[c] = 1

	target_space  = VectorFunctionSpace(mesh, 'DG', 2, 5)
	control_space = TensorFunctionSpace(mesh, 'DG', 2, (5, 5))
	base_space    = FunctionSpace(mesh, 'DG', 2)


	pspeed_0 = 1600.
	sspeed_0 = 800.
	rho_0    = 1100.

	pspeed_1 = 4000.
	sspeed_1 = 2000.
	rho_1    = 1800.

	def calculate_lame_params(pspeed, sspeed, rho):
		return (pspeed ** 2 - 2. * sspeed ** 2) * rho, sspeed ** 2 * rho 

	lambda_0, mu_0 = calculate_lame_params(pspeed_0, sspeed_0, rho_0)
	lambda_1, mu_1 = calculate_lame_params(pspeed_1, sspeed_1, rho_1)


	ctrls_mu      = Function(base_space)
	ctrls_lambda  = Function(base_space)
	ctrls_rho     = Function(base_space)

        
	# set constant values

	ctrls_lambda.assign(interpolate(lambda_(subdomains, lambda_0, lambda_1, degree=2), base_space))
	ctrls_mu.assign(interpolate(mu_(subdomains, mu_0, mu_1, degree=2), base_space))
	ctrls_rho.assign(interpolate(rho_(subdomains, rho_0, rho_1, degree=2), base_space))


	A = Function(control_space)
	assigner_A = FunctionAssigner(
    	control_space, [base_space] * control_space.num_sub_spaces())
	
	comps_A = [
	    *[
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        project(-ctrls_lambda - 2 * ctrls_mu, base_space),
	        interpolate(Constant(0.0), base_space)
	    ],
	    *[
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        project(-ctrls_lambda, base_space),
	        interpolate(Constant(0.0), base_space),
	    ],
	    *[
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        project(-ctrls_mu, base_space)
	    ],
	    *[
	        project(-1. / ctrls_rho, base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space)
	    ],
	    *[
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        project(-1. / ctrls_rho, base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space)
	    ]    
	]

	assigner_A.assign(A, comps_A)


	B = Function(control_space)
	assigner_B = FunctionAssigner(control_space, [base_space] * control_space.num_sub_spaces())

	comps_B = [
	    *[
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        project(-ctrls_lambda, base_space),
	    ],
	    *[
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        project(-ctrls_lambda - 2 * ctrls_mu, base_space),
	    ],
	    *[
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        project(-ctrls_mu, base_space),
	        interpolate(Constant(0.0), base_space)
	    ],
	    *[
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        project(-1. / ctrls_rho, base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space)
	    ],    
	    *[
	        interpolate(Constant(0.0), base_space),
	        project(-1. / ctrls_rho, base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space),
	        interpolate(Constant(0.0), base_space)
	    ],   
	]

	assigner_B.assign(B, comps_B)

	T = 1.0 # final time
	num_steps = 100 # number of time steps
	dt = T / num_steps # time step size

	vtkfile_vx = File(
    	os.path.join(os.getcwd(), 'snapshots', 'vx.pvd')) 
	vtkfile_vy = File(
    	os.path.join(os.getcwd(), 'snapshots', 'vy.pvd')) 


	tol = 1e-14

	class boundary_top(SubDomain):
	    def inside(self, x, on_boundary): return on_boundary and near(x[1], 2000., tol)
	    
	class boundary_bot(SubDomain):
	    def inside(self, x, on_boundary): return on_boundary and near(x[1], 0., tol)

	class boundary_left(SubDomain):
	    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0., tol)

	class boundary_right(SubDomain):
	    def inside(self, x, on_boundary): return on_boundary and near(x[0], 2000., tol)
	    
	top   = boundary_top()
	bot   = boundary_bot()
	left  = boundary_left()
	right = boundary_right()

	# test with dirichlet boundary condition
	# TODO: free boundary?

	boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
	left.mark(boundaries, 1)
	right.mark(boundaries, 2)
	bot.mark(boundaries, 3)
	top.mark(boundaries, 4)

	# gaussian functor
	f = Expression(
	    (
	        "0.0",
	        "0.0",
	        "0.0",
	        "1. / 100 * exp(-a * (pow(x[0] - x_0, 2) + pow(x[1] - x_1, 2))) * exp(-b * pow(t, 2)) * sin(2. * 10. * t)",
	        "1. / 100 * exp(-a * (pow(x[0] - x_0, 2) + pow(x[1] - x_1, 2))) * exp(-b * pow(t, 2)) * sin(2. * 10. * t)"
	    ), degree=2,
	    x_0=1000.0,
	    x_1=500.0, 
	    t=0., 
	    a=1 / 100000., 
	    b=100.
	)

	gauss_functor = DirichletBC(
	    target_space, f, top
	)

	free_bot_1 = DirichletBC(target_space.sub(0), Constant(0.), bot)
	free_bot_2 = DirichletBC(target_space.sub(1), Constant(0.), bot)
	free_bot_3 = DirichletBC(target_space.sub(2), Constant(0.), bot)

	free_left_1 = DirichletBC(target_space.sub(0), Constant(0.), left)
	free_left_2 = DirichletBC(target_space.sub(1), Constant(0.), left)
	free_left_3 = DirichletBC(target_space.sub(2), Constant(0.), left)

	free_right_1 = DirichletBC(target_space.sub(0), Constant(0.), right)
	free_right_2 = DirichletBC(target_space.sub(1), Constant(0.), right)
	free_right_3 = DirichletBC(target_space.sub(2), Constant(0.), right)



	BC = [
	    free_bot_1,
	    free_bot_2,
	    free_bot_3,
	    free_left_1,
	    free_left_2,
	    free_left_3,
	    free_right_1,
	    free_right_2,
	    free_right_3
	]

	u_n = interpolate(
	    Expression(("0.0", "0.0", "0.0", "0.0", "0.0"), degree=2),
	    target_space
	)

	u = TrialFunction(target_space)
	v = TestFunction(target_space)

	F = dot(u, v) * dx +\
	        dt * dot((A * grad(u)[:, 0] + B * grad(u)[:, 1]), v) * dx -\
	        (dot(u_n, v)) * dx
	    
	a, L = lhs(F), rhs(F)
	    
	t = 0
	u = Function(target_space)

	vtkfile_vx << (u_n.sub(3), t)
	vtkfile_vy << (u_n.sub(4), t)
	    
	for n in range(num_steps):
	    # Update current time
	    t += dt
	    f.t = t
	    # Compute solution
	    solve(a == L, u, BC)
	    u_n.assign(u)
	    
	    #if n % 10 == 0:        
	    vtkfile_vx << (u_n.sub(3), t)
	    vtkfile_vy << (u_n.sub(4), t)

	J = assemble(dot(u_n, u_n)*dx)

	control = Control(A)

	adj_timer = Timer("Adjoint run")
	dJdm = compute_gradient(J, control) 
	adj_time = adj_timer.stop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_file" , help="xml file containing the mesh", type=str)
    parser.add_argument("mesh_tags_file", help="xml file with mesh markup")
    
    args   = parser.parse_args()

    mesh_file = args.mesh_file
    mesh_tags = args.mesh_tags_file

    main(mesh_file, mesh_tags)
