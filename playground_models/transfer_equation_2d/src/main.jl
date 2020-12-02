include("blackpaper.jl")
using DelimitedFiles

# predefined hyperparameters for solution

# center location

const x_00 = 0.0
const y_00 = 0.0

# velocity field radius

const r_1 = 1.0

# center movement hyperparameters

const u_00 = 1.0 # transfer speed for center
const r_00 = 0.1 # radius of the circle where the center of the initial condition travels
const T_0  = 1.0  # time of the single rotation of the initial condition

# hyperparameters for transfer speeds

const T_1 = 0.5 # time of the single rotation around the center
const T_2 = 0.5 # Decaying factor

# hyperparameters for initial conditions

const H_0   = 1.0 # height of the cone
const r_0   = 0.5 # radius of the cone
#angles defining the slice of the cone
const phi_0 = pi / 4.0 
const phi_1 = pi / 2.0


# define new fuctions from templates
# to build the solution, we need a single sample of
# initial conditions, transfer speeds, and center point movement functions


# radius-vector from central point defined above
function r(x, y)
	return r_template(x, y, x_00, y_00)
end 

# x coordinate for center point
function x0_linear(t)
	return x0_linear_template(t, u_00, x_00)
end

# y coordinate for center point
function y0_linear(t)
	return x0_linear_template(t, u_00, y_00)
end

function a1_linear(r)
	return r
end

function a2_constant(t)
	return a2_constant_template(t, T_1)
end

# initial conditions in polar coordinates 
function cone_polar(r, phi)
	return cone_template(r, phi, r_0, H_0)
end

function cone_slice_polar(r, phi)
	return cone_slice_template(r, phi, r_0, H_0, phi_0, phi_1)
end


# initial conditions in cartesian coordinates
function cone_cartesian(xs, ys)

	grid = [(x, y) for x in xs, y in ys]
	
	radius = inflate(
			(x, y) -> ((x .- x_00) ^ 2 + (y .- y_00) ^ 2) ^ 0.5,
			xs, ys
	)

	rlh = [
		r != 0.0 ?
		acos((x .- x_00) / r) :
		0.5 * pi
		for (r, (x, y)) in zip(radius, grid)
	]

	rrh = [
		r != 0.0 ? 
		(2.0 * pi .- acos((x .- x_00) / r)) :
		1.5 * pi
		for  (r, (x, y)) in zip(radius, grid)
	]

	phi     = zeros((length(xs), length(ys)))
	mask    = [y .<  y_00 for (x, y) in grid]
	notmask	= [y .>= y_00 for (x, y) in grid]
	phi[mask]    = rrh[mask]
	phi[notmask] = rlh[notmask]

	periods = floor.(phi / (2.0 * pi))
	phi     = phi - periods * 2.0 * pi

	return cone_template(radius, phi, r_0, H_0)


function cone_slice_cartesian(xs, ys)

	grid = [(x, y) for x in xs, y in ys]
	
	radius = inflate(
			(x, y) -> ((x .- x_00) ^ 2 + (y .- y_00) ^ 2) ^ 0.5,
			xs, ys
	)

	rlh = [
		r != 0.0 ?
		acos((x .- x_00) / r) :
		0.5 * pi
		for (r, (x, y)) in zip(radius, grid)
	]

	rrh = [
		r != 0.0 ? 
		(2.0 * pi .- acos((x .- x_00) / r)) :
		1.5 * pi
		for  (r, (x, y)) in zip(radius, grid)
	]

	phi     = zeros((length(xs), length(ys)))
	mask    = [y .<  y_00 for (x, y) in grid]
	notmask	= [y .>= y_00 for (x, y) in grid]
	phi[mask]    = rrh[mask]
	phi[notmask] = rlh[notmask]

	periods = floor.(phi / (2.0 * pi))
	phi     = phi - periods * 2.0 * pi

	return cone_slice_template(r, phi, r_0, H_0, phi_0, phi_1)


# transfer velocities
#function v1(x, y, t)
#	return v1_template(x, y, t, )


function solution(x, y, t)

	return solution_template(
		x, y, t, 
		cone_slice,
		a1_linear,
		a2_constant,
		x0_linear,
		y0_linear
	)

end


# setup the domain

x = -1.0:0.01:1.0
y = -1.0:0.01:1.0
t = 0.0:0.1:10

res = solution(x, y, t)

# save as .csv file

writedlm("result.csv", res, ',')