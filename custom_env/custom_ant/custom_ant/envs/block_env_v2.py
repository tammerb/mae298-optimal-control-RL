import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py as mjp


DEFAULT_CAMERA_CONFIG = {
    'distance': 6.0,
}


class BlockV2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file= os.getcwd() + '/..' + '/custom_ant/models/block_two_legs_v0.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        
        self.num_timesteps = 0
        self.target = [5,5]

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        
        xy_position_before = self.get_body_com("torso")[:2].copy()
        block_position_before = self.get_body_com("block")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        self.num_timesteps += self.frame_skip
        block_position_after = self.get_body_com("block")[:2].copy()
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        block_velocity = (block_position_after - block_position_before) / self.dt
        block_x_velocity, block_y_velocity = block_velocity
        target_dis = self.target - block_position_after
        
        mjp.functions.mj_rnePostConstraint(self.sim.model, self.data) #### calc contacts        

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        
        
         # normalization 
        norm = np.linalg.norm(block_velocity)
        xy_velo_norm = block_velocity/norm
        x_comp,y_comp = xy_velo_norm
        theta= np.arctan2(y_comp,x_comp)* 180 / np.pi
        
        # Desired goal
        x_desired = 5
        y_desired = 7
        theta_desired = np.arctan2(y_desired,x_desired)* 180 / np.pi
        
        # Distance Calculations
        x_before, y_before = block_position_before
        x_after, y_after = block_position_after
        
        D_after = ((y_desired - y_after)**2 + (x_desired - x_after)**2)**0.5
        D_before = ((y_desired - y_before)**2 + (x_desired - x_before)**2)**0.5
        D_diff = D_before - D_after
        D_compare = abs(D_after)
    
        #forward_reward = 5 * block_x_velocity + 0.1 * xy_position_after[0]
        #forward_reward = 1/abs(theta_desired - theta) 
        
        
        # 100*D_diff -> If D_after is greater than D_before, meaning it is getting further away from the goal, D_diff is negative ( negative reward )
        #            -> Likewise, if the ant model is approaching the goal, reward is positive 
        
        forward_reward = 10*D_diff + 1/(D_after)**2 
        healthy_reward = self.healthy_reward

        #print(forward_reward)
        rewards =  healthy_reward + forward_reward 
        # rewards = (100*forward_reward + healthy_reward)*0.01


        #print(rewards)
        costs = ctrl_cost  + contact_cost + np.absolute(block_y_velocity)
        #costs = contact_cost
        # testing constact force 
        # contact_forces_test = self.data.get_sensor('torsoSensor') 
        #contact_forces_test = self.data.sensordata 
        #print(contact_forces_test)
        #print(' ')

        reward = rewards - costs
        #reward = rewards
        if D_after <= 1:
            done = True
            reward += 2000
        elif self.num_timesteps > 5000:
            done = True
            self.num_timesteps = 0
        else:
            done = self.done
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity, contact_force))

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
