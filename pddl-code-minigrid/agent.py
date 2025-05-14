class Agent():
    def __init__(self):
        self.current_dir = ""
        self.current_observation = [[]]
        self.full_grid = [[]]
        self.previous_actions = []

    # predicates
    def am_next_to(self, obj):
        if self.current_observation[0,0+1] == obj and self.current_dir == "right":
            return True
        return False
    
    # actions
    def turn_left(self):
        return 0
    def turn_right(self):
        return 1
    def move_forward(self):
        return 2
    def pick_up(self):
        return 3
    
    # new actions
    def safe_forward(self):
        if not self.lava_ahead():
            return self.move_forward()
    
    def pick_up_obj(self, obj):
        if self.am_next_to(obj):
            return self.pick_up()
        
