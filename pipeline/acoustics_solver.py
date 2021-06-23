import numpy as np
import os


def step_source(t):
    # return 0.0
    # return np.exp(-0.5 * ((t - 0.005) / 0.0001) ** 2) if t < 0.01 else 0.0
    #return np.sin((np.pi / 0.01) * t) ** 4 if t < 0.01 else 0.0
    return 100.0 if t < 0.01 else 0.0


class acoustics_solver():

    def __init__(self, x_size, y_size, cp, rho, target_time, recording_time_step,
                 source_width, source_function = step_source,
                 dump_vtk = False, dump_dir = "data", verbose = False):

        assert (cp.shape == rho.shape)

        self.dump_vtk = dump_vtk
        self.dump_dir = dump_dir
        self.verbose = verbose

        self.x_size = x_size
        self.y_size = y_size
        self.num_points_x, self.num_points_y = cp.shape
        self.hx = self.x_size / (self.num_points_x - 1)
        self.hy = self.y_size / (self.num_points_y - 1)

        assert (abs(self.hx - self.hy) < 0.01 * (self.hx + self.hy))

        self.cp = cp
        self.rho = rho
        self.K = np.square(self.cp) * self.rho

        self.T = 0

        max_cp = np.max(self.cp)
        numerical_method_recommended_tau = 0.5 * min(self.hx / max_cp, self.hy / max_cp)

        if self.verbose:
            print("Numerical time step recommendation:", numerical_method_recommended_tau)

        self.number_of_records = int(target_time / recording_time_step)
        self.steps_per_record = max(int(recording_time_step / numerical_method_recommended_tau), 1)
        self.tau = recording_time_step / self.steps_per_record

        if self.verbose:
            print("Doing %d data records, %d steps per record, total %d steps, time step is %f, final time %f" %
                  (self.number_of_records, self.steps_per_record,
                   self.number_of_records * self.steps_per_record, self.tau, target_time))

        self.x = np.linspace(0.0, self.x_size, self.num_points_x, endpoint=True)
        self.y = np.linspace(0.0, self.y_size, self.num_points_y, endpoint=True)
        self.z = np.array([0.0])

        if self.dump_vtk:
            from pyevtk.hl import gridToVTK
            gridToVTK(os.path.join(self.dump_dir, "params"), self.x, self.y, self.z,
                      pointData = {"Cp" : self.cp.T.ravel(), "rho" : self.rho.T.ravel(), "K" : self.K.T.ravel()})

        source_half_width_in_points = int(source_width / (2 * self.hx))
        self.source_start_point = (self.num_points_x // 2) - source_half_width_in_points
        self.source_end_point = (self.num_points_x // 2) + source_half_width_in_points

        self.source = source_function
        if self.verbose:
            print("Grid shape", cp.shape)
            print("The source is located from %d to %d" % (self.source_start_point, self.source_end_point))


    def forward(self):
        if self.dump_vtk:
            from pyevtk.hl import gridToVTK

        # We are going to use GCM for 2D acoustics.
        # The math, magic and matrices used below
        # are described in details in https://keldysh.ru/council/3/D00202403/kazakov_ao_diss.pdf

        Ax = np.zeros((self.num_points_x, self.num_points_y, 3, 3))
        Ax[:,:,0,2] = np.power(self.rho, -1)
        Ax[:,:,2,0] = self.K

        Ay = np.zeros((self.num_points_x, self.num_points_y, 3, 3))
        Ay[:,:,1,2] = np.power(self.rho, -1)
        Ay[:,:,2,1] = self.K

        Ux = np.zeros((self.num_points_x, self.num_points_y, 3, 3))
        Ux[:,:,0,0] = 1
        Ux[:,:,0,2] = np.power(self.cp * self.rho, -1)
        Ux[:,:,1,0] = 1
        Ux[:,:,1,2] = - np.power(self.cp * self.rho, -1)
        Ux[:,:,2,1] = 2
        Ux = Ux / np.sqrt(2)

        Ux1 = np.zeros((self.num_points_x, self.num_points_y, 3, 3))
        Ux1[:,:,0,0] = 1
        Ux1[:,:,0,1] = 1
        Ux1[:,:,1,2] = 1
        Ux1[:,:,2,0] = self.cp * self.rho
        Ux1[:,:,2,1] = - self.cp * self.rho
        Ux1 = Ux1 / np.sqrt(2)

        Uy = np.zeros((self.num_points_x, self.num_points_y, 3, 3))
        Uy[:,:,0,1] = 1
        Uy[:,:,0,2] = np.power(self.cp * self.rho, -1)
        Uy[:,:,1,1] = 1
        Uy[:,:,1,2] = - np.power(self.cp * self.rho, -1)
        Uy[:,:,2,0] = 2
        Uy = Uy / np.sqrt(2)

        Uy1 = np.zeros((self.num_points_x, self.num_points_y, 3, 3))
        Uy1[:,:,0,2] = 1
        Uy1[:,:,1,0] = 1
        Uy1[:,:,1,1] = 1
        Uy1[:,:,2,0] = self.cp * self.rho
        Uy1[:,:,2,1] = - self.cp * self.rho
        Uy1 = Uy1 / np.sqrt(2)

        # Initial state
        u_prev = np.zeros((self.num_points_x, self.num_points_y, 3))
        # The response recorded will be stored here
        buffer = np.zeros((self.num_points_x, self.number_of_records))

        if self.dump_vtk:
            gridToVTK(os.path.join(self.dump_dir, "u" + str(0)), self.x, self.y, self.z,
                      pointData={"vx": u_prev[:,:,0].T.ravel(),
                                 "vy": u_prev[:,:,1].T.ravel(),
                                 "p": u_prev[:,:,2].T.ravel()})

        # Spacial steps for characterictics (they are constant, since the model is linear).
        steps = self.tau * self.cp

        # We are going to use these formulas - https://math.stackexchange.com/a/889571 - for interpolation

        # positive direction
        c1p = steps * (- self.hx + steps) / (2 * self.hx * self.hx)
        c2p = (self.hx + steps) * (- self.hx + steps) / (- self.hx * self.hx)
        c3p = (self.hx + steps) * steps / (2 * self.hx * self.hx)

        # negative direction
        c1m = steps * (self.hx + steps) / (2 * self.hx * self.hx)
        c2m = (- self.hx + steps) * (self.hx + steps) / (- self.hx * self.hx)
        c3m = (- self.hx + steps) * steps / (2 * self.hx * self.hx)

        # Time stepping using space-split scheme.
        # TODO: the formulas below are monkey-coded a bit, but let's refactor them later (if required)
        for i in range(self.number_of_records):
            if self.verbose:
                print("Stepping for the time record", i)

            for j in range(self.steps_per_record):
                if self.verbose:
                    print("Step", j)

                ### X step

                u_lefts = np.pad(u_prev, ((1, 0),(0,0),(0,0)), mode='constant', constant_values=0)[:-1,:,:]
                u_rights = np.pad(u_prev, ((0, 1),(0,0),(0,0)), mode='constant', constant_values=0)[1:,:,:]

                rieman_invs_here = np.einsum('qijk,qik->qij', Ux, u_prev)
                rieman_invs_left = np.einsum('qijk,qik->qij', Ux, u_lefts)
                rieman_invs_right = np.einsum('qijk,qik->qij', Ux, u_rights)

                limiter_min = np.min([rieman_invs_here, rieman_invs_left, rieman_invs_right], axis=0)
                limiter_max = np.max([rieman_invs_here, rieman_invs_left, rieman_invs_right], axis=0)

                rieman_invs_pos = c1m * rieman_invs_left[:,:, 0] + c2m * rieman_invs_here[:,:, 0] + c3m * rieman_invs_right[:,:, 0]
                rieman_invs_neg = c1p * rieman_invs_left[:,:, 1] + c2p * rieman_invs_here[:,:, 1] + c3p * rieman_invs_right[:,:, 1]
                rieman_invs_zero = rieman_invs_here[:,:, 2]

                riemans_next = np.zeros((self.num_points_x, self.num_points_y, 3))
                riemans_next[:,:, 0] = rieman_invs_pos
                riemans_next[:,:, 1] = rieman_invs_neg
                riemans_next[:,:, 2] = rieman_invs_zero

                riemans_next = np.max([riemans_next, limiter_min], axis=0)
                riemans_next = np.min([riemans_next, limiter_max], axis=0)

                u_next = np.einsum('qijk,qik->qij', Ux1, riemans_next)

                ### Y step

                u_bottoms = np.pad(u_next, ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)[:,:-1,:]
                u_tops = np.pad(u_next, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)[:,1:,:]

                rieman_invs_here = np.einsum('qijk,qik->qij', Uy, u_next)
                rieman_invs_left = np.einsum('qijk,qik->qij', Uy, u_bottoms)
                rieman_invs_right = np.einsum('qijk,qik->qij', Uy, u_tops)

                limiter_min = np.min([rieman_invs_here, rieman_invs_left, rieman_invs_right], axis=0)
                limiter_max = np.max([rieman_invs_here, rieman_invs_left, rieman_invs_right], axis=0)

                rieman_invs_pos = c1m * rieman_invs_left[:,:, 0] + c2m * rieman_invs_here[:,:, 0] + c3m * rieman_invs_right[:,:, 0]
                rieman_invs_neg = c1p * rieman_invs_left[:,:, 1] + c2p * rieman_invs_here[:,:, 1] + c3p * rieman_invs_right[:,:, 1]
                rieman_invs_zero = rieman_invs_here[:,:, 2]

                riemans_next = np.zeros((self.num_points_x, self.num_points_y, 3))
                riemans_next[:,:, 0] = rieman_invs_pos
                riemans_next[:,:, 1] = rieman_invs_neg
                riemans_next[:,:, 2] = rieman_invs_zero

                riemans_next = np.max([riemans_next, limiter_min], axis=0)
                riemans_next = np.min([riemans_next, limiter_max], axis=0)

                u_next = np.einsum('qijk,qik->qij', Uy1, riemans_next)

                # emitting and reflecting top border
                form = self.source(self.T)
                p0 = np.zeros(self.num_points_x)
                p0[self.source_start_point: self.source_end_point] = form
                u_next[:,-1,2] = p0
                u_next[:,-1,1] = riemans_next[:,0,1] + p0 / (self.cp[:,0] * self.rho[:,0])

                buffer[:, -i-1] = np.copy(u_next[:,-2,2])

                if self.dump_vtk:
                    gridToVTK(os.path.join(self.dump_dir, "u" + str(i + 1)), self.x, self.y, self.z,
                                           pointData={"vx": u_next[:,:,0].T.ravel(),
                                                      "vy": u_next[:,:,1].T.ravel(),
                                                      "p": u_next[:,:,2].T.ravel()})

                u_prev = np.copy(u_next)

                self.T += self.tau

        return buffer.T