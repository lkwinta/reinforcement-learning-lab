# python plot.py --output raport_imgs \
#     "results/reach_her_future_alpha_0.05/sac_her_log.json:HERFuture" \
#     "results/reach_her_final_alpha_0.05/sac_her_log.json:HERFinal" \
#     "results/reach_her_episode_alpha_0.05/sac_her_log.json:HEREpisode" \

# python plot.py --output raport_imgs \
#     "results/reach_her_future_alpha_0.05/sac_her_log.json:HERFutureBuffer" \
#     "results/reach_her_future_alpha_auto/sac_her_log.json:HERFutureAutoAlpha"

# python plot.py --output raport_imgs \
#     "results/reach_her_future_alpha_auto/sac_her_log.json:her_n_sampled_goal=3" \
#     "results/reach_her_future_alpha_auto_6/sac_her_log.json:her_n_sampled_goal=6"

# python plot.py --output "pick_and_place_her_future_alpha_auto/plots" "pick_and_place_her_future_alpha_auto/sac_her_log.json:HERFutureBufferAuto"

# python plot.py --output push_her_future_alpha_auto/plots \
#     "push_her_future_alpha_auto/sac_her_log.json:HERFutureAutoAlpha"

# python plot.py --output pick_her_future_alpha_auto/plots \
#     "pick_her_future_alpha_auto/sac_her_log.json:HERFutureAutoAlpha"

python plot.py --output pick_her_future_alpha_auto_256/plots \
    "pick_her_future_alpha_auto_256/sac_her_log.json:HERFutureAutoAlpha256"
