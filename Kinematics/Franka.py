import numpy as np
import RobotUtil as rt
import math

class FrankArm:
    def __init__(self):
        # Robot descriptor taken from URDF file (rpy xyz for each rigid link transform) - NOTE: don't change
        self.Rdesc = [
            [0, 0, 0, 0., 0, 0.333],  # From robot base to joint1
            [-np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0, -0.316, 0],
            [np.pi/2, 0, 0, 0.0825, 0, 0],
            [-np.pi/2, 0, 0, -0.0825, 0.384, 0],
            [np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0.088, 0, 0],
            [0, 0, 0, 0, 0, 0.107]  # From joint5 to end-effector center
        ]

        # Define the axis of rotation for each joint
        self.axis = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]

        # Set base coordinate frame as identity - NOTE: don't change
        self.Tbase = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

        # Initialize matrices - NOTE: don't change this part
        self.Tlink = []  # Transforms for each link (const)
        self.Tjoint = []  # Transforms for each joint (init eye)
        self.Tcurr = []  # Coordinate frame of current (init eye)
        
        for i in range(len(self.Rdesc)):
            self.Tlink.append(rt.rpyxyz2H(
                self.Rdesc[i][0:3], self.Rdesc[i][3:6]))
            self.Tcurr.append([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 1, 0.], [0, 0, 0, 1]])
            self.Tjoint.append([[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 1, 0.], [0, 0, 0, 1]])

        self.Tlinkzero = rt.rpyxyz2H(self.Rdesc[0][0:3], self.Rdesc[0][3:6])

        self.Tlink[0] = np.matmul(self.Tbase, self.Tlink[0])

        # initialize Jacobian matrix
        self.J = np.zeros((6, 7))

        self.q = [0., 0., 0., 0., 0., 0., 0.]
        self.ForwardKin([0., 0., 0., 0., 0., 0., 0.])

    def ForwardKin(self, ang):
        '''
        inputs: joint angles
        outputs: joint transforms for each joint, Jacobian matrix
        '''

        
        self.q[0:-1] = ang

        for i in range(len(self.Rdesc)):
            self.Tjoint[i] = rt.rpyxyz2H(self.q[i]*np.array(self.axis[i]), np.array([0, 0, 0]))
            
            if(i == 0):
                self.Tcurr[i] = np.array(self.Tlink[i])@np.array(self.Tjoint[i])
            else:
                self.Tcurr[i] = np.array(self.Tcurr[i-1])@(np.array(self.Tlink[i])@np.array(self.Tjoint[i]))
        
        for i in range(len(self.Rdesc)-1):
            self.J[:, i] = np.vstack((np.cross(self.Tcurr[i][:3, 2], (self.Tcurr[-1][:3, 3] - self.Tcurr[i][:3, 3])).reshape(3,1), self.Tcurr[i][:3, 2].reshape(3,1))).T
            
    
        return np.array(self.Tcurr), np.array(self.J)

    def IterInvKin(self, ang, TGoal, x_eps=1e-3, r_eps=1e-3):
        '''
        inputs: starting joint angles (ang), target end effector pose (TGoal)

        outputs: computed joint angles to achieve desired end effector pose, 
        Error in your IK solution compared to the desired target
        '''

        W = np.eye(7)
        C = np.eye(6)

        
        
        C = [[1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0],
             [0.0, 0.0,   0.0,  1000.0, 0.0, 0.0],
             [0.0, 0.0,   0.0,  0.0, 1000.0, 0.0],
             [0.0, 0.0,   0.0,  0.0, 0.0, 1000.0]]
        
        W = [[1.0, 0.0, 0.0,   0.0, 0.0, 0.0,   0.0],
             [0.0, 1.0, 0.0,   0.0, 0.0, 0.0,   0.0],
             [0.0, 0.0, 100.0, 0.0, 0.0, 0.0,   0.0],
             [0.0, 0.0, 0.0, 100.0, 0.0, 0.0,   0.0],
             [0.0, 0.0, 0.0,   0.0, 1.0, 0.0,   0.0],
             [0.0, 0.0, 0.0,   0.0, 0.0, 1.0,   0.0],
             [0.0, 0.0, 0.0,   0.0, 0.0, 0.0, 100.0]]
        
        self.ForwardKin(ang)

        W_inv = np.linalg.inv(W)
        C_inv = np.linalg.inv(C) 

        Terr_lim = 1e-1
        Rerr_lim = 1e-2    
        R_err = np.array((3, 1), dtype=float)
        T_err = np.array((3,1), dtype=float)

        Itr_ = 0
        while (x_eps < np.linalg.norm(T_err) or r_eps < np.linalg.norm(R_err)):

            # Get Rotation Error
            R_err = TGoal[:3, :3]@np.transpose(self.Tcurr[-1][:3, :3])
            R_axis, R_angle = rt.R2axisang(R_err)


            if R_angle > Rerr_lim:
                R_err = Rerr_lim*np.array(R_axis)
            else:
                R_err = R_angle*np.array(R_axis)
            
            # Get Translational Error
            T_err = TGoal[:3, 3] - self.Tcurr[-1][:3, 3]
            
            if np.linalg.norm(T_err) > Terr_lim :
                T_err = T_err*Terr_lim    
            

            Err = np.concatenate((T_err, R_err), axis=0).reshape(6,1)
            
            J_pinv = W_inv@np.transpose(self.J)@np.linalg.inv(self.J@W_inv@np.transpose(self.J) + C_inv) 
            
            self.q[0:-1] +=  np.squeeze(J_pinv@Err)

            self.ForwardKin(self.q[0:-1])
            Itr_+=1
            # print(f" Itr: {Itr_}| Translation Err :{np.linalg.norm(T_err)} | Rotation Err :{np.linalg.norm(R_err)} ")

        # print(f"Converged at at Iteration :{Itr_}")
        return self.q[0:-1], Err
