import torch
import dmc2gym
from logger import Logger
from video import VideoRecorder

from curl_sac import RadSacAgent



def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel' and 'crop' in args.data_augs:
                    obs = utils.center_crop_image(obs,args.image_size)
                if args.encoder_type == 'pixel' and 'translate' in args.data_augs:
                    # first crop the center with pre_image_size
                    obs = utils.center_crop_image(obs, args.pre_transform_image_size)
                    # then translate cropped to center
                    obs = utils.center_translate(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs / 255.)
                    else:
                        action = agent.select_action(obs / 255.)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

        filename = args.work_dir + '/' + args.domain_name + '--'+args.task_name + '-' + args.data_augs + '--s' + str(args.seed) + '--eval_scores.npy'
        key = args.domain_name + '-' + args.task_name + '-' + args.data_augs
        try:
            log_data = np.load(filename,allow_pickle=True)
            log_data = log_data.item()
        except:
            log_data = {}
            
        if key not in log_data:
            log_data[key] = {}

        log_data[key][step] = {}
        log_data[key][step]['step'] = step 
        log_data[key][step]['mean_ep_reward'] = mean_ep_reward 
        log_data[key][step]['max_ep_reward'] = best_ep_reward 
        log_data[key][step]['std_ep_reward'] = std_ep_reward 
        log_data[key][step]['env_step'] = step * args.action_repeat

        np.save(filename,log_data)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)

def set_env():
  env = dmc2gym.make(
      domain_name='cartpole',
      task_name='swingup',
      visualize_reward=False,
      from_pixels=True,
  )

  return env

def load_agent(env):
  action_shape = env.action_space.shape
  obs_shape = env.observation_space.shape
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  agent = RadSacAgent(obs_shape, action_shape, device, data_augs='no_aug', training=False)
  agent.load('tmp/cartpole-swingup-11-10-im84-b128-s23-pixel/model', 10000)

  return agent

if __name__ == '__main__':
  env = set_env()
  agent = load_agent(env)

