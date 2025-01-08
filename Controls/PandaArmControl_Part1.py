import mujoco as mj
from mujoco import viewer
import numpy as np
import math
import quaternion
import time as t

# Set the XML filepath
xml_filepath = "../franka_emika_panda/panda_nohand_torque_fixed_board.xml"

start_time_ = t.time()
ImpedanceFlag = False
ForceFlag = False        
PositionFlag = True
    
################################# Control Callback Definitions #############################

# Control callback for gravity compensation
def gravity_comp(model, data):
    # data.ctrl exposes the member that sets the actuator control inputs that participate in the
    # physics, data.qfrc_bias exposes the gravity forces expressed in generalized coordinates, i.e.
    # as torques about the joints

    data.ctrl[:7] = data.qfrc_bias[:7]

# Force control callback
def force_control(model, data):  # TODO:
    # Implement a force control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    # Instantite a handle to the desired body on the robot

    end_eff = data.body("hand")

    # Get the Jacobian for the desired location on the robot (The end-effector)    
    # This function works by taking in return parameters!!! Make sure you supply it with placeholder
    # variables
    jacP = np.zeros((3,model.nv))
    jacR = np.zeros((3,model.nv))

    mj.mj_jac(model, data, jacP, jacR, end_eff.xpos, end_eff.id)
    
    Jacob = np.vstack((jacP, jacR))

    # Specify the desired force in global coordinates
    Fdes_x = 15
    
    F_des = np.array([Fdes_x, 0, 0, 0, 0, 0])
    
    
    # Compute the required control input using desied force values
    tau_des = np.matmul(np.transpose(Jacob),F_des)
    
    # Set the control inputs
    data.ctrl[:7] = tau_des + data.qfrc_bias[:7]

    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Force readings updated here
    # Update force sensor readings
    # force[:] = np.roll(force, -1)[:]
    # force[-1] = data.sensordata[2]
    # print(f"Elapsed Time :{(t.time() - start_time_)}")
    if((t.time() - start_time_) > 10.00):
        # force[:] = np.roll(force, -1)[:]
        # force[-1] = data.sensordata[2]
        print("Not updating force anymore")
    else:
        force[:] = np.roll(force, -1)[:]
        force[-1] = data.sensordata[2]


# Control callback for an impedance controller
def impedance_control(model, data):  # TODO:

    # Implement an impedance control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    # Instantite a handle to the desired body on the robot
    end_eff = data.body("hand")
    # Current Pos and Velocity 
    pose_curr = np.concatenate((end_eff.xpos, np.array([0, 0, 0])))
    vel_curr = end_eff.cvel

    # Desired Pos and Velocity
    board = data.body("whiteboard")
    pos_des = np.concatenate((board.xpos, np.array([0, 0, 0])))
    vel_des = board.cvel


    # Pos and Vel Err
    pose_err = pos_des - pose_curr
    vel_err = vel_des - vel_curr

    
    # Get the Jacobian at the desired location on the robot
    jacP = np.zeros((3,model.nv))
    jacR = np.zeros((3,model.nv))

    mj.mj_jac(model, data, jacP, jacR, end_eff.xpos, end_eff.id)
    
    Jacob = np.vstack((jacP, jacR))
    

    # Get Motor torques for gravity
    f_g  = data.qfrc_bias[:7]

    
    # Kp = 100
    # Kp = 15/np.linang.norm(pose_err)
    # print(f"Pose err :{np.linang.norm(pose_err)}")
    # print(f"Kp :{Kp}")
    # Kd = 0.01

    # Hack for 15N control
    Kp = 1
    pose_err = np.array([15, 0, 0, 0, 0, 0])
    Kd = 0.01
    # Compute the required control input using desied force values
    
    Imp = Kp*(pose_err)  + Kd*(vel_err)
    # print(f"Impedance :{np.linalg.norm(Imp)}")
    tau_des = np.matmul(np.transpose(Jacob),Imp)

    # This function works by taking in return parameters!!! Make sure you supply it with placeholder
    # variables

    # Compute the impedance control input torques
    
    # Set the control inputs
    data.ctrl[:7] = f_g + tau_des

    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Update force sensor readings
    # force[:] = np.roll(force, -1)[:]
    # force[-1] = data.sensordata[2]
    # print(f"Elapsed Time :{(t.time() - start_time_)}")
    if((t.time() - start_time_) > 10.00):
        # force[:] = np.roll(force, -1)[:]
        # force[-1] = data.sensordata[2]
        print("Not updating force anymore")
    else:
        force[:] = np.roll(force, -1)[:]
        force[-1] = data.sensordata[2]


def position_control(model, data):
    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Set the desired joint angle positions
    desired_joint_positions = np.array(
        [0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
    
    # Values from 
    
    desired_joint_positions = np.array([0.6861859276510833, 0.34630181018357475, -
     0.8233220687997146, -2.075870299687756, -0.14744290658976544, 3.919305788784955, -2.7831659162987696])
    

    # Set the desired joint velocities
    desired_joint_velocities = np.array([0, 0, 0, 0, 0, 0, 0])

    # Desired gain on position error (K_p)
    Kp = 1000

    # Desired gain on velocity error (K_d)
    Kd = 1000

    # Set the actuator control torques
    data.ctrl[:7] = data.qfrc_bias[:7] + Kp * \
        (desired_joint_positions-data.qpos[:7]) + Kd * \
        (np.array([0, 0, 0, 0, 0, 0, 0])-data.qvel[:7])

    
####################################### MAIN #####################################
if __name__ == "__main__":
    # Load the xml file here
    model = mj.MjModel.from_xml_path(xml_filepath)
    data = mj.MjData(model)

    # Set the simulation scene to the home configuration
    mj.mj_resetDataKeyframe(model, data, 0)

    ################################# Swap Callback Below This Line #################################
    # This is where you can set the control callback. Take a look at the Mujoco documentation for more
    # details. Very briefly, at every timestep, a user-defined callback function can be provided to
    # mujoco that sets the control inputs to the actuator elements in the model. The gravity
    # compensation callback has been implemented for you. Run the file and play with the model as
    # explained in the PDF

    if PositionFlag == True:
        mj.set_mjcb_control(position_control)
    elif ForceFlag == True:
        mj.set_mjcb_control(force_control)
    elif ImpedanceFlag == True:   
        mj.set_mjcb_control(impedance_control)
    
    # mj.set_mjcb_control(gravity_comp)  # TODO:

    ################################# Swap Callback Above This Line #################################

    # Initialize variables to store force and time data points
    force_sensor_max_time = 10
    force = np.zeros(int(force_sensor_max_time/model.opt.timestep))
    time = np.linspace(0, force_sensor_max_time, int(
        force_sensor_max_time/model.opt.timestep))
    
    viewer.launch(model, data)


    # # Save recorded force and time points as a csv file
    force = np.reshape(force, (5000, 1))
    time = np.reshape(time, (5000, 1))
    plot = np.concatenate((time, force), axis=1)
    if PositionFlag == True:
        np.savetxt('position_vs_time.csv', plot, delimiter=',')
    elif ForceFlag == True:
        np.savetxt('force_vs_time.csv', plot, delimiter=',')
    elif ImpedanceFlag == True:   
        np.savetxt('impedance_vs_time.csv', plot, delimiter=',')

    
    
