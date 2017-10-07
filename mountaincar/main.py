from mountaincar.gymhelpers import ExperimentsManager

env_name = "MountainCar-v0"
root_dir = '/media/diego/Data/relearning/'
results_dir_prefix = root_dir +env_name
figures_dir = root_dir +'Figures'
checkpoints_dir = root_dir +'GymCheckP'
summaries_path = root_dir +'TensorBoardSummaries'
api_key = 'your_key'

n_ep = 3500
n_exps = 1

hidden_layers_size = [198, 96]

expsman = ExperimentsManager(env_name=env_name, results_dir_prefix=results_dir_prefix, summaries_path=summaries_path,
                             agent_value_function_hidden_layers_size=hidden_layers_size, figures_dir=figures_dir,
                             discount=0.99, decay_eps=0.99, eps_min=1E-4, learning_rate=2.33e-4, decay_lr=False,
                             max_step=200, replay_memory_max_size=100000, ep_verbose=False, exp_verbose=True,
                             batch_size=64, upload_last_exp=True, double_dqn=False,
                             target_params_update_period_steps=999, gym_api_key=api_key,
                             checkpoints_dir=checkpoints_dir)
expsman.run_experiments(n_exps=n_exps, n_ep=n_ep, stop_training_min_avg_rwd=-97, plot_results=False)

input("Press Enter to terminate.")
