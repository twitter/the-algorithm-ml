This project is the "heavy ranker" used on the "For You" timeline. This is used to generate the ranking of Tweet after candidate retrieval and light ranker (note the final ordering of the Tweet is not directly the highest -> lowest scoring, because after scoring other heuristics are used).

This model captures the ranking model used for the majority of users of Twitter "For You" timeline in early March 2023. Due to the need to make sure this runs independently from other parts of Twitter codebase, there may be small differences from the production model.

The model receives various features, describing the Tweet and the user whose timeline is being constructed as input (see FEATURES.md for more details). The model outputs multiple binary predictions about how the user will respond if shown the Tweet.


Those are:
-  "recap.engagement.is_favorited": The probability the user will favorite the Tweet.
-  "recap.engagement.is_good_clicked_convo_desc_favorited_or_replied": The probability the user will click into the conversation of this Tweet and reply or Like a Tweet.
-  "recap.engagement.is_good_clicked_convo_desc_v2": The probability the user will click into the conversation of this Tweet and stay there for at least 2 minutes.
-  "recap.engagement.is_negative_feedback_v2": The probability the user will react negatively (requesting "show less often" on the Tweet or author, block or mute the Tweet author)
-  "recap.engagement.is_profile_clicked_and_profile_engaged": The probability the user opens the Tweet author profile and Likes or replies to a Tweet.
-  "recap.engagement.is_replied": The probability the user replies to the Tweet.
-  "recap.engagement.is_replied_reply_engaged_by_author": The probability the user replies to the Tweet and this reply is engaged by the Tweet author.
-  "recap.engagement.is_report_tweet_clicked": The probability the user will click Report Tweet.
-  "recap.engagement.is_retweeted": The probability the user will ReTweet the Tweet.
-  "recap.engagement.is_video_playback_50": The probability (for a video Tweet) that the user will watch at least half of the video

For ranking the candidates these predictions are combined into a score by weighting them:
-  "recap.engagement.is_favorited": 0.5
-  "recap.engagement.is_good_clicked_convo_desc_favorited_or_replied": 11* (the maximum prediction from these two "good click" features is used and weighted by 11, the other prediction is ignored).
-  "recap.engagement.is_good_clicked_convo_desc_v2": 11*
-  "recap.engagement.is_negative_feedback_v2": -74
-  "recap.engagement.is_profile_clicked_and_profile_engaged": 12
-  "recap.engagement.is_replied": 27
-  "recap.engagement.is_replied_reply_engaged_by_author": 75
-  "recap.engagement.is_report_tweet_clicked": -369
-  "recap.engagement.is_retweeted": 1
-  "recap.engagement.is_video_playback_50": 0.005


We cannot release the real training data due to privacy restrictions. However, we have included a script to generate random data to ensure you can run the model training code.

To try training the model (assuming you have already followed the repo setup instructions and are inside a virtualenv).

Run
```
$ ./projects/home/recap/scripts/create_random_data.sh
```

This will create some random data (in $HOME/tmp/recap_local_random_data).
```
$ ./projects/home/recap/scripts/run_local.sh
```

This will train the model (for a small number of iterations). Checkpoints and logs will be written to $HOME/tmp/runs/recap_local_debug.

The model training is configured through a yaml file (./projects/home/recap/config/local_prod.yaml).

The model architecture is a parallel masknet (https://arxiv.org/abs/2102.07619).
