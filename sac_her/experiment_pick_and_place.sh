python train_pick.py --logs_dir "pick_her_future_alpha_auto" --alpha auto_1.0 --buffer_type her --her_strategy future --her_n_sampled_goal 4 \
     --n_steps 3000000
# python train_pick.py --logs_dir "pick_her_future_alpha_auto_256" --alpha auto_1.0 --buffer_type her --her_strategy future --her_n_sampled_goal 4 \
#      --n_steps 1000000

python plot.py --output "pick_her_future_alpha_auto/plots" "pick_her_future_alpha_auto/sac_her_log.json:HERFutureAutoAlpha"
# python plot.py --output "pick_her_future_alpha_auto_256/plots" "pick_her_future_alpha_auto_256/sac_her_log.json:HERFutureAutoAlpha256"

# python plot.py --output plots_pick \
#     "pick_her_future_alpha_auto/sac_her_log.json:HERFutureAutoAlpha" \
#     "pick_her_future_alpha_auto_256/sac_her_log.json:HERFutureAutoAlpha256"
