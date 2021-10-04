import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os

# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
sim_settings = {
    "scene": "Replica/apartment_0/habitat/mesh_semantic.ply",  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

def make_simple_cfg(settings):
    """ This function generates a config for the simulator.
        It contains two parts:
        one for the simulator backend
        one for the agent, where you can attach a bunch of sensors
        Args:
            settings ([type]): [description]

        Returns:
            [type]: [description]
    """
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
   
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # Attach a RGB sensor to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # Attach a depth snesor to the agent
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # Attach a semantic snesor to the agent
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


class InteractiveAgent:
    def __init__(self, args):
        self.__dict__.update(args.__dict__)
        position = [0.0, 0.0, 0.0] if self.floor == 'first' else [0.0, 1.0, 0.0]
        cfg = make_simple_cfg(sim_settings)
        self.sim = habitat_sim.Simulator(cfg)
        # initialize an agent
        self.agent = self.sim.initialize_agent(sim_settings["default_agent"])
        # Set agent state
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(position)  # agent in world space
        self.agent.set_state(agent_state)

        # obtain the default, discrete actions that an agent can perform
        # default action space contains 3 actions: move_forward, turn_left, and turn_right
        self.action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
        print("Supported actions: ", self.action_names)

        self.seri_num = 0
        self.pose_record = []
        if self.save:
            self.images_dir = os.path.join(self.save_dir, 'images')
            os.makedirs(self.images_dir, exist_ok=True)


    def visualize_color(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    def visualize_depth(self, image):
        return (image / 10 * 255).astype(np.uint8)


    def visualize_label(self, semantic_obs):
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")
        semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)

        return semantic_img


    def step_and_record(self, action):
        assert action in self.action_names

        observations = self.sim.step(action)
        color = observations["color_sensor"]
        depth = observations["depth_sensor"] 
        label = observations["semantic_sensor"]

        cv2.imshow("color", self.visualize_color(color))
        if self.vis_depth:
            cv2.imshow("depth", self.visualize_depth(depth))
        if self.vis_label:
            cv2.imshow("label", self.visualize_label(label))

        agent_state = self.agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        pos = sensor_state.position.astype(np.float64)
        rot = sensor_state.rotation.components
        print("camera pose: [x y z] [rw rx ry rz]")
        print(pos, rot)

        if self.save and not (self.only_forward and action != "move_forward"):
            path_templet = os.path.join(self.images_dir, '%03d-{}.png' %self.seri_num)
            cv2.imwrite(path_templet.format('color'), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            cv2.imwrite(path_templet.format('depth'), (depth * self.dscale).astype(np.uint16))
            # cv2.imwrite(path_templet.format('label'), label.astype(np.uint16))
            
            self.pose_record.append(np.hstack((pos, rot)))
            self.seri_num+=1


    def start(self):
        print("#############################")
        print("use keyboard to control the agent")
        print(" w for go forward  ")
        print(" a for turn left  ")
        print(" d for trun right  ")
        print(" f for finish and quit the program")
        print("#############################")

        self.step_and_record("move_forward")
        while True:
            keystroke = cv2.waitKey(0)
            if keystroke == ord('w'):
                self.step_and_record("move_forward")
                print("action: FORWARD")
            elif keystroke == ord('a'):
                self.step_and_record("turn_left")
                print("action: LEFT")
            elif keystroke == ord('d'):
                self.step_and_record("turn_right")
                print("action: RIGHT")
            elif keystroke == ord('q'):
                print("action: FINISH")
                break
            else:
                print("INVALID KEY")
                continue

        if self.save:
            pose_path = os.path.join(self.save_dir, 'GT_pose.txt')
            np.savetxt(pose_path, np.vstack(self.pose_record), fmt='%10.5f')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default="Data_collection")
    parser.add_argument('--sub_dir')
    parser.add_argument('--floor', default='first', choices=['first', 'second'])
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--only_forward', action='store_true', help="Only save images when moving forward")
    parser.add_argument('--dscale', default=4000, type=int, help="depth scale")
    parser.add_argument('--vis_depth', action='store_true')
    parser.add_argument('--vis_label', action='store_true')

    args = parser.parse_args()
    if args.sub_dir is not None:
        args.save_dir = os.path.join(args.save_dir, args.sub_dir)

    my_agent = InteractiveAgent(args)
    my_agent.start()