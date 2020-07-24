import numpy as np
from numpy.linalg import inv

class KalmanModel:

    def __init__(self, args, acceleration=False):
        self.obs_len = args.obs_length
        self.pred_len = args.pred_length
        
        self.p0 = 15#15
        self.q = .03#0.03
        self.r = .03#0.03
        self.data_fps = 2
        self.dt = 1/self.data_fps
        self.A, self.H = self.init_matrices(acceleration, dt = self.dt)
        self.m_dim, self.n_dim = self.H.shape
        
        self.Q = np.diag(np.full(self.n_dim, self.q))# Process Noise
        self.R = np.diag(np.full(self.m_dim, self.r))# Observation Noise
        self.P0 = np.diag(np.full(self.n_dim, self.p0))# Covariance matrix
        self.K = np.zeros((self.m_dim, self.n_dim))
        self.P = None
    
    def init_matrices(self, acceleration, dt=0.1):
        if acceleration:  # use acceleration
            # transition matrix  x  x'  y  y'  x'' y''
            A = np.array([[1, 1 * dt, 0, 0, 0.5 * dt * dt, 0],  # x
                        [0, 1, 0, 0, 1 * dt, 0],  # x'
                        [0, 0, 1, 1 * dt, 0, 0.5 * dt * dt],  # y
                        [0, 0, 0, 1, 0, 1 * dt],  # y'
                        [0, 0, 0, 0, 1, 0],  # x''
                        [0, 0, 0, 0, 0, 1]])  # y''
                        # 6x6

            H = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0]])# 2x6
        else:
            # transition matrix x  x' y  y'
            A = np.array([[1, 1 * dt, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1 * dt],
                        [0, 0, 0, 1]])

            H = np.array([[1, 0, 0, 0],  # m x n     m = 2, n = 4 or 6
                        [0, 0, 1, 0]])
        return A, H
        
    def PredictTraj(self, trajectory):
        ''' 
        Kalman
        : The whole system consists of calculating the filter parameter through n observation data
            Then, using that parameter, it is going to estimate the k frames trajectory!
        '''

        x_hat_previous = None

        # Get agent's 
        # agents_obs_data: Contains [x,y,type] of each agent in each history' frame.
        #                  the last frame is 'frame_idx' which means the current frame
        pred_trajectory = np.empty((self.pred_len, self.n_dim))#n_dim could be (x, x', y, y') or (x, x', y, y', x'', y'')
        self.K = np.zeros((self.m_dim, self.n_dim))
        self.P = self.P0

        if len(trajectory) <= 1:
            return None
        
        # State initialization.
        x0, y0 = trajectory[0]
        
        v0x = (trajectory[1, 0] - x0) / self.dt#velocity
        v0y = (trajectory[1, 1] - y0) / self.dt
        if self.n_dim == 6:
            x_hat_previous = np.array([x0, v0x, y0, v0y, 0, 0])
        elif self.n_dim == 4:
            x_hat_previous = np.array([x0, v0x, y0, v0y])
        covariance_list = [None]*self.pred_len

        # Take the first n frames and fit the kalman filter on them.
        for j, z_k in enumerate(trajectory):#range(len(trajectory)):
            x_hat_new = self.fit(x_hat_previous, z_k)  # predict and correct
            x_hat_previous = x_hat_new

        for u in range(self.pred_len):
            x_hat_new = self.predict(x_hat_previous)
            covariance_list[u] = self.P
            pred_trajectory[u] = x_hat_new
            x_hat_previous = x_hat_new

        return pred_trajectory, covariance_list

    def fit(self, xhat_previous, z_k):
        """
        main iteration: we need the state estimate x_hat_minus k-1 at previous step and the current measurement z_k

        :param xhat_previous: previous a posteriori prediction
        :param z_k: current measurement: (tx,ty) tuple
        :return: new a posteriori prediction
        """

        # prediction
        xhat_k_minus = self.predict(xhat_previous)  # predict updates self.P (P_minus)
        P_minus = self.P

        # innovation
        residual = z_k - np.dot(self.H, xhat_k_minus)
        inv_HPHTR = inv(np.dot(np.dot(self.H, P_minus), self.H.T) + self.R)

        # correction (update)
        self.K = np.dot(np.dot(P_minus, self.H.T), inv_HPHTR)# Calculate Kalman gain
        xhat_k_new = xhat_k_minus + np.dot(self.K, residual)
        self.P = np.dot((np.eye(self.n_dim) - np.dot(self.K, self.H)), P_minus)

        return xhat_k_new

    def predict(self, xhat_k_previous):
        xhat_k_minus = np.dot(self.A, xhat_k_previous)  # update previous state estimate with state transition matrix
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q  # this is P minus; Covariacne

        return xhat_k_minus
