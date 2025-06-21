from turtle import done
import gymnasium as gym
from gymnasium import spaces
from screen import Display
from swiplserver import PrologMQI, PrologThread
import random

class BlocksWorldEnvTarget(gym.Env):
    RENDER_FPS = 60  # Frames per second for rendering
    metadata = {"render_modes": ["human"], "render_fps": RENDER_FPS}

    def __init__(self, render_mode=None):
        super().__init__()

        # a. Start PrologMQI and load blocks_world_with_target.pl
        self.render_mode = render_mode
        self.mqi = PrologMQI()
        self.prolog_thread = self.mqi.create_thread()
        result = self.prolog_thread.query("[blocks_world_with_target]")
        if not result:
            raise RuntimeError("Failed to load blocks_world_with_target.pl")

        # b. Get all states and build state -> int mapping/ Prolog State -> index
        self.states_dict = {}
        state_results = self.prolog_thread.query('state(State)')
        for i, item in enumerate(state_results):
            self.states_dict[item['State']] = i
        self.inv_states_dict = {v: k for k, v in self.states_dict.items()}

        # Print states dict for debugging
        print("States Dictionary:", self.states_dict)

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
        print("Actions Dictionary:", self.actions_dict)
        
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

        # b. Get current 3-digit agent state
        result = self.prolog_thread.query('current_state(State)')
        agent_state_str = result[0]['State']

        # c. Randomly pick a 3-digit target (from state_helper states)
        target_state_str = random.choice([
            state[:3] for state in self.states_dict.keys()
        ])
        

    
        # d. Create the 6-digit state
        full_state_str = agent_state_str + target_state_str
        self.state = self.states_dict[full_state_str]
        self.target = self.states_dict[full_state_str]  # keep it consistent

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
        
        # c. Prolog returns the 3-digit new agent state
        result = self.prolog_thread.query("current_state(State)")
        agent_state_str = result[0]['State']

        # d. Append the stored 3-digit target
        target_str = self.inv_states_dict[self.target][3:]  # last 3 digits
        full_state_str = agent_state_str + target_str
        self.state = self.states_dict[full_state_str]

        done = self.state == self.target
        reward = 100 if done else -1

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

        full_state = self.inv_states_dict[self.state]
        agent = full_state[:3]
        target = full_state[3:]

        # draw state
        self.display.draw(agent)
        self.display.target = target
        self.display.step(agent)

    def close(self):
        if hasattr(self, 'display') and getattr(self.display, 'screen', None) is not None:
            self.display.close_window()
        self.mqi.stop()