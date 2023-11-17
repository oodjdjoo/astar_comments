#Swarm Flight Test in Airsim
#23.11.07 ~ 23.11.10
# (C) Kaveh Fathian, 2017-2018.

import setup_path 
import airsim

import os
import time
import numpy as np
import pprint
import time


# Total n UAVs
numUAV = 8

pos0 = np.zeros((numUAV,8))
for i in range(numUAV):
    pos0[i,0] = 588.0 + 4.0*i               #setting.json X Coordinate
    pos0[i,1] = -28.0 + 4.0*i       #setting.json Y Coordinate
    pos0[i,2] = -125.0              #setting.json Z Coordinate


# Connect AirSim Simulator (UE4_SwarmFlight)

client = airsim.MultirotorClient()
client.confirmConnection()



for i in range(numUAV):
    name = "UAV" + str(i+1)
    client.enableApiControl(True, name)
    client.armDisarm(True, name)
print("All UAVs have been Binding")


# Hovering

time.sleep(2) 

tout = 2 # Timeout in seconds
spd = 0.3 # Speed 

airsim.wait_key('Press any key to Hovering')

print("taking off...")
for i in range(numUAV):    
    name = "UAV" + str(i+1)
    print('Hovering', name)
    client.hoverAsync(vehicle_name = name)
    client.moveToPositionAsync(0, 0, -9000, spd, timeout_sec = tout, vehicle_name = name)
print("All UAVs are hovering.")



# Move all Drone Position (Test_OneWay)
# X = -10, Y = 10, Z = -10, Spd = 3m/s


airsim.wait_key('Press any key to move vehicle to (-10, 10, -100) at 3 m/s')
print("move drone position")


for i in range(numUAV):
    name = "UAV" + str(i + 1)

    # UAV 1~3 Coordinate
    if i <= 3:
        client.hoverAsync(vehicle_name=name)
        client.moveToPositionAsync(-10, 10, -150000, spd, timeout_sec=tout, vehicle_name=name)
        print('Move', name)
    # UAV 3~10 Coordinate
    elif i > 3:
        client.hoverAsync(vehicle_name=name)
        client.moveToPositionAsync(-10, 10, -120000, spd, timeout_sec=tout, vehicle_name=name)
        print('Move', name)
    # Additional condition for UAV 7~8 Coordinate
    elif 7 <= i <= 8:
        client.hoverAsync(vehicle_name=name)
        client.moveToPositionAsync(-10, 10, -100000, spd, timeout_sec=tout, vehicle_name=name)
        print('Move', name)
  

#client.moveAsync(vehicle_name = name)
#client.moveToPositionAsync(-10, 10, -100, spd, timeout_sec = tout, vehicle_name = name)

print("All UAVs are hovering.")


# Client Reset

airsim.wait_key('Press any key to Reset Client')
client.reset()