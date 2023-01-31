from djitellopy import tello
import time as time

drone = tello.Tello()
drone.connect()

print("Battery:" ,drone.get_battery(),"%")

# finding the height for checking base
print("Altitude: ",drone.get_height(),"cm")

# drone.takeoff()
# for i in range(3):
#     print("Altitude: ",drone.get_height(),"cm")
#     time.sleep(1)

# drone.land()


