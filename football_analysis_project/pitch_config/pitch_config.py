class SoccerPitchConfiguration:
    def __init__(self):
        # FIFA Standard dimensions in meters
        self.width = 68.0
        self.length = 105.0
        self.penalty_box_width = 40.32
        self.penalty_box_length = 16.5
        self.goal_box_width = 18.32
        self.goal_box_length = 5.5
        self.centre_circle_radius = 9.15
        self.penalty_spot_distance = 11.0
        
        # 32 exact keypoints expected by the Roboflow-trained YOLO model
        self.vertices = [
            [0, 0],  # 1: Top Left Corner
            [0, (self.width - self.penalty_box_width) / 2],  # 2
            [0, (self.width - self.goal_box_width) / 2],  # 3
            [0, (self.width + self.goal_box_width) / 2],  # 4
            [0, (self.width + self.penalty_box_width) / 2],  # 5
            [0, self.width],  # 6: Bottom Left Corner
            [self.goal_box_length, (self.width - self.goal_box_width) / 2],  # 7
            [self.goal_box_length, (self.width + self.goal_box_width) / 2],  # 8
            [self.penalty_spot_distance, self.width / 2],  # 9: Penalty Spot
            [self.penalty_box_length, (self.width - self.penalty_box_width) / 2],  # 10
            [self.penalty_box_length, (self.width - self.goal_box_width) / 2],  # 11
            [self.penalty_box_length, (self.width + self.goal_box_width) / 2],  # 12
            [self.penalty_box_length, (self.width + self.penalty_box_width) / 2],  # 13
            [self.length / 2, 0],  # 14: Halfway line Top
            [self.length / 2, self.width / 2 - self.centre_circle_radius],  # 15
            [self.length / 2, self.width / 2 + self.centre_circle_radius],  # 16
            [self.length / 2, self.width],  # 17: Halfway line Bottom
            [self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2],  # 18
            [self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2],  # 19
            [self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2],  # 20
            [self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2],  # 21
            [self.length - self.penalty_spot_distance, self.width / 2],  # 22
            [self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2],  # 23
            [self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2],  # 24
            [self.length, 0],  # 25: Top Right Corner
            [self.length, (self.width - self.penalty_box_width) / 2],  # 26
            [self.length, (self.width - self.goal_box_width) / 2],  # 27
            [self.length, (self.width + self.goal_box_width) / 2],  # 28
            [self.length, (self.width + self.penalty_box_width) / 2],  # 29
            [self.length, self.width],  # 30: Bottom Right Corner
            [self.length / 2 - self.centre_circle_radius, self.width / 2],  # 31
            [self.length / 2 + self.centre_circle_radius, self.width / 2],  # 32
        ]