
python train_push.py --logs_dir "push_her_future_alpha_auto" --alpha auto_1.0 --buffer_type her --her_strategy future --her_n_sampled_goal 4 \
     --n_steps 1000000

python plot.py --output "push_her_future_alpha_auto/plots" "push_her_future_alpha_auto/sac_her_log.json:HERFutureAutoAlpha"
