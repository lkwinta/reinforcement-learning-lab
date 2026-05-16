python train_reach.py --logs_dir "reach_dict_alpha_0.05" --alpha 0.05 --buffer_type dict
python train_reach.py --logs_dir "reach_her_future_alpha_0.05" --alpha 0.05 --buffer_type her --her_strategy future --her_n_sampled_goal 3
python train_reach.py --logs_dir "reach_her_final_alpha_0.05" --alpha 0.05 --buffer_type her --her_strategy final --her_n_sampled_goal 3
python train_reach.py --logs_dir "reach_her_episode_alpha_0.05" --alpha 0.05 --buffer_type her --her_strategy episode --her_n_sampled_goal 3
python train_reach.py --logs_dir "reach_her_future_alpha_0.05_6" --alpha 0.05 --buffer_type her --her_strategy future --her_n_sampled_goal 6
python train_reach.py --logs_dir "reach_her_future_alpha_auto" --alpha auto_0.5 --buffer_type her --her_strategy future --her_n_sampled_goal 3
python train_reach.py --logs_dir "reach_her_future_alpha_auto_6" --alpha auto_0.5 --buffer_type her --her_strategy future --her_n_sampled_goal 6

python plot.py --output "reach_dict_alpha_0.05/plots" "reach_dict_alpha_0.05/sac_her_log.json:DictBuffer"
python plot.py --output "reach_her_future_alpha_0.05/plots" "reach_her_future_alpha_0.05/sac_her_log.json:HERFutureBuffer"
python plot.py --output "reach_her_final_alpha_0.05/plots" "reach_her_final_alpha_0.05/sac_her_log.json:HERFinalBuffer"
python plot.py --output "reach_her_episode_alpha_0.05/plots" "reach_her_episode_alpha_0.05/sac_her_log.json:HEREpisodeBuffer"
python plot.py --output "reach_her_future_alpha_0.05_6/plots" "reach_her_future_alpha_0.05_6/sac_her_log.json:HERFutureBuffer6"
python plot.py --output "reach_her_future_alpha_auto/plots" "reach_her_future_alpha_auto/sac_her_log.json:HERFutureAutoAlpha"
python plot.py --output "reach_her_future_alpha_auto_6/plots" "reach_her_future_alpha_auto_6/sac_her_log.json:HERFutureAutoAlpha6"

python plot.py --output plots_reach \
    "reach_dict_alpha_0.05/sac_her_log.json:DictBuffer" \
    "reach_her_future_alpha_0.05/sac_her_log.json:HERFutureBuffer" \
    "reach_her_final_alpha_0.05/sac_her_log.json:HERFinalBuffer" \
    "reach_her_episode_alpha_0.05/sac_her_log.json:HEREpisodeBuffer" \
    "reach_her_future_alpha_0.05_6/sac_her_log.json:HERFutureBuffer6" \
    "reach_her_future_alpha_auto/sac_her_log.json:HERFutureAutoAlpha" \
    "reach_her_future_alpha_auto_6/sac_her_log.json:HERFutureAutoAlpha6"

