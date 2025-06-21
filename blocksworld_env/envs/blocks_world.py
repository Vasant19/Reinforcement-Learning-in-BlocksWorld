import sys
import gymnasium as gym
from gymnasium import spaces
from screen import Display
from swiplserver import PrologMQI, PrologThread
import random

class BlocksWorldEnv(gym.Env):
    RENDER_FPS = 60  # Frames per second for rendering
    metadata = {"render_modes": ["human"], "render_fps": RENDER_FPS}

    def __init__(self, render_mode=None):
        super().__init__()

        # a. Start PrologMQI and load blocks_world.pl
        self.render_mode = render_mode
        self.mqi = PrologMQI()
        self.prolog_thread = self.mqi.create_thread()
        result = self.prolog_thread.query("[blocks_world]")
        if not result:
            raise RuntimeError("Failed to load blocks_world.pl")
        
        # b. Get all states and build state -> int mapping/ Prolog State -> index
        self.states_dict = {}
        state_results = self.prolog_thread.query('state(State)')
        for i, item in enumerate(state_results):
            self.states_dict[item['State']] = i
        self.inv_states_dict = {v: k for k, v in self.states_dict.items()}

        # Print states dict for debugging
        # print("States Dictionary:", self.states_dict)

        # c. Create action dictionary: int -> Prolog action string
        self.actions_dict = {}
        result = self.prolog_thread.query("action(A)")
        # result is like: [{'A': {'args': ['a', 'b', 'c'], 'functor': 'move'}},...]
        for i, A in enumerate(result):
            action_string = A['A']['functor']
            first = True
            for arg in A['A']['args']:
                if first:
                    first = False
                    action_string += '('
                else:
                    action_string += ','
                action_string += str(arg)
            action_string += ')'
            self.actions_dict[i] = action_string  # maps index to action string
        # Print actions dict for debugging
        # print("Actions Dictionary:", self.actions_dict)
        
        # d. Define observation and action space
        self.observation_space = spaces.Discrete(len(self.states_dict))
        self.action_space = spaces.Discrete(len(self.actions_dict))

        # e. Initial state and target
        self.state = list(self.states_dict.values())[0]
        self.target = list(self.states_dict.values())[1]
        
        # f. Render mode
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.display = Display()
        else:
            self.display = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # a. Reset the Prolog environment
        self.prolog_thread.query('reset.')

        # b. Get the current state string after reset
        result = self.prolog_thread.query('current_state(State)')
        state_str = result[0]['State']

        # c. Convert to integer state ID using our state dictionary
        self.state = self.states_dict[state_str]

        # d. Optional: randomly pick a new goal state
        self.target = random.choice(list(self.states_dict.values()))

        # e. Return initial observation and info dict (optional goal state)
        return self.state, {"target": self.target}

    def step(self, action):
        # a. Convert action index to Prolog term string
        action_str = self.actions_dict[action]

        # b. Run the action in Prolog
        success = self.prolog_thread.query(f"step({action_str}).")
        
        # Check if the action was successful
        if not success:
            reward = -10
            done = False
            return self.state, reward, done, False, {"target": self.target}
        
        # c. Valid move: Get new state string from Prolog
        result = self.prolog_thread.query("current_state(State)")
        new_state_str = result[0]['State']
        self.state = self.states_dict[new_state_str]

        # d. Check if we reached the goal
        done = self.state == self.target
        reward = 100 if done else -1

        # Debugging Output
        # print(f"Action taken: {action_str}")
        # print(f"New state: {new_state_str} -> {self.state}")

        # e. Return the Gym-compatible tuple
        return self.state, reward, done, False, {"target": self.target}
    
    def render(self, mode="human"):
        if self.render_mode != "human":
            # Skip rendering for other modes or None
            return
        
        if not hasattr(self, 'display') or self.display is None:
            self.display = Display()
        
        if self.display.screen is None:
            self.display.__init__()  # re-init pygame display

        # draw current state
        self.display.draw(self.inv_states_dict[self.state])  # pass current state string
        self.display.target = self.inv_states_dict[self.target]
        self.display.step(self.inv_states_dict[self.state])

    def close(self):
        if self.display:
            try:
                self.display.close_window()
            except Exception as e:
                print("Warning: Failed to close display:", e)

        if hasattr(self, 'prolog_thread') and self.prolog_thread:
            try:
                self.prolog_thread.stop()
            except Exception as e:
                print("Warning: Failed to stop PrologThread:", e)

        if hasattr(self, 'mqi') and self.mqi:
            try:
                self.mqi.stop()
                print("PrologMQI stopped successfully.")
            except Exception as e:
                print("Warning: Failed to stop PrologMQI:", e)