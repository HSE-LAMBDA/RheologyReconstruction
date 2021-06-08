#=
Author: Stankevich Andrey, MIPT, Russia, 2020

This package provides function templates for 
exact solutions of the 2D transfer equation
Cauchy problem 
See https://keldysh.ru/papers/2002/prep4/prep2002_4.html
=#

# helper function
inflate(f, xs, ys) = [f(x, y) for x in xs, y in ys]


####################
# radius vector norm

function r_template(x, y, x_00, y_00)
	return ((x .- x_00) .^ 2 + (y .- y_00) .^ 2) .^ 0.5
end


#################
# center location

function x0_constant_template(t, x_00)
	return x_00 * ones(size(t))
end

function dx0_dt_constant(t)
	return zeros(size(t))
end

function y0_constant_template(t, y_00)
	return y_00 * ones(size(t))
end

function dy0_dt_constant(t)
	return zeros(size(t))
end

function x0_linear_template(t, u_00, x_00)
	return x_00 .+ u_00 * t
end

function dx0_dt_linear_template(t, u_00)
	return u_00 * ones(size(t))
end

function y0_linear_template(t, u_00, y_00)
	return y_00 .+ u_00 * t
end

function dy0_dt_linear_template(t, u_00)
	return u_00 * ones(size(t))
end

function x0_circle_template(t, r_00, T_0)
	return r_00 * cos(2.0 * pi * t / T_0)
end

function dx0_dt_circle_template(t, r_00, T_0)
	return (2.0 * pi / T_0) * r_00 * sin(2.0 * pi * t / T_0) * (-1.0)
end

function y0_circle_template(t, r_00, T_0)
	return r_00 * sin(2.0 * pi * t / T_0)
end

function dy0_dt_circle_template(t, r_00, T_0)
	return (2.0 * pi / T_0) * r_00 * cos(2.0 * pi * t / T_0)
end


############################################
# parametric function a2 for transfer speeds

function a2_constant_template(t, T_1)
	return 2.0 * π * t / T_1 
end 

function da2_dt_constant_template(t, T_1)
	return 2.0 * π / T_1
end

function da2_increasing_template(t, T_1, c)
	return 2.0 * π * (t + c * t^2) / (T_1 + c * T_1^2)
end

function da2_dt_increasing_template(t, T_1, c)
	return 2.0 * π * (2 * c * t) /(T_1 + c * T_1^2)
end

function da2_decaying_template(t, T_1, T_2)
	return 2.0 * π * (1 - exp(-t/T_1)) / (1 - exp(-T_1/T_2))
end

function da2_dt_decaying_template(t, T_1, T_2)
	return 2.0 * π * exp(-t/T_1) / (1 - exp(-T_1/T_2))
end


############################################
# parametric function a1 for transfer speeds

function a1_linear(radius)
	return radius
end 

function da1_dt_linear(radius)
	return radius .* 0
end


####################
# Initial conditions

function cone_template(r, phi, r_1, H_0)
	return H_0 * clamp.(1.0 .- r / r_1, 0.0, 1.0)
end

function cone_slice_template(r, phi, r_1, H_0, phi_0, phi_1)

	result = H_0 * clamp.(1.0 .- r / r_1, 0.0, 1.0)
	mask   = map(&, phi .<= phi_1, phi .>= phi_0)
	
	result[mask] .= 0.0

	return result
end 


#############################
# transfer velocity functions
# should have a specific form

function v1_template(x, y, t, a1, da2_dt, y0, dx0_dt, r1)

	radius = r(x, y)

	lh = a1(radius) * (y .- y0(t)) / radius * da2_dt(t) + dx0_dt(t)

	return lh * (radius .<= r1)
end


function v2_template(x, y, t, a1, da1_dt, x0, dy0_dt, r1)
	
	radius = r(x, y)
	lh = -a1(radius) * (x .- x0(t)) / radius * da1_dt(t) + dy0_dt(t)

	return lh * (radius .<= r1)
end


#############################
# solution


import Base.Iterators.product

function solution_template(xs, ys, ts, R, a1, a2, x0, y0)

	grid = [(x, y) for x in xs, y in ys]

	result = []

	for t in ts

		x0_coord = x0(t)
		y0_coord = y0(t)

		radius = inflate(
			(x, y) -> ((x .- x0_coord) ^ 2 + (y .- y0_coord) ^ 2) ^ 0.5,
			xs, ys
		)


		rlh = [
			r != 0.0 ?
			acos((x .- x0_coord) / r) :
			0.5 * pi
			for (r, (x, y)) in zip(radius, grid)
		]

		rrh = [
			r != 0.0 ? 
			(2.0 * pi .- acos((x .- x0_coord) / r)) :
			1.5 * pi
			for  (r, (x, y)) in zip(radius, grid)
		]

		
		rh   = zeros((length(xs), length(ys)))
		mask    = [y .<  y0_coord for (x, y) in grid]
		notmask	= [y .>= y0_coord for (x, y) in grid]
		rh[mask]    = rrh[mask]
		rh[notmask] = rlh[notmask]


		phi = [r != 0.0 ? a1(r) * a2(t) / r : 0.0 for r in radius] 
		phi = phi + rh

		periods = floor.(phi / (2.0 * pi))
		phi     = phi - periods * 2.0 * pi

		push!(result, R(radius, phi))

	end
	
	return result

end


	






