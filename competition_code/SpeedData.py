class SpeedData:
    TTH_THRESHOLD = { # threshold for minimum time to hit before CAS kicks in
        0: 0.01,
        1: 0.01,
        2: 0.01,
        3: 0.1,
        4: 0.01,
        5: 0.1,
        6: 0.01,
        7: 0.01,
        8: 0.01,
        9: 0.01
    }

    def __init__(
        self, distance_to_section, current_speed, target_speed, recommended_speed
    ):
        self.current_speed = current_speed
        self.distance_to_section = distance_to_section
        self.target_speed_at_distance = target_speed
        self.recommended_speed_now = recommended_speed
        self.speed_diff = current_speed - recommended_speed
