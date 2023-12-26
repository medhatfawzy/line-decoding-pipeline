import airsim
from processing import Driver


if __name__ == "__main__":
    # connect to the AirSim simulator
    client = airsim.CarClient()
    
    # car controls object
    car_controls = airsim.CarControls()
    
    # get state of the car
    car_state = client.getCarState()
    
    client.confirmConnection()
    client.enableApiControl(True)

    print(f"API Control enabled: {client.isApiControlEnabled()}")
    print(f"Speed {car_state.speed}, Gear {car_state.gear}")

    # print(f"Car state: {car_state}")
    driver = Driver(client, car_controls)
    driver.drive(save_input=True)