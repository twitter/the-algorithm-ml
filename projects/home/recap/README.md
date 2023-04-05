# Heavy Ranker

## Overview

The heavy ranker is a machine learning model used to rank tweets for the "For You" timeline
which have passed through the candidate retrieval stage. It is one of the final stages of the funnel, 
succeeded primarily by a set of filtering heuristics.

The model receives features describing a Tweet and the user that the Tweet is being recommended to 
(see [FEATURES.md](./FEATURES.md)). The model architecture is a parallel [MaskNet](https://arxiv.org/abs/2102.07619) 
which outputs a set of numbers between 0 and 1, with each output representing the probability that the user 
will engage with the tweet in a particular way. The predicted engagement types are explained below:
```
scored_tweets_model_weight_fav: The probability the user will favorite the Tweet.
scored_tweets_model_weight_retweet: The probability the user will Retweet the Tweet.
scored_tweets_model_weight_reply: The probability the user replies to the Tweet.
scored_tweets_model_weight_good_profile_click: The probability the user opens the Tweet author profile and Likes or replies to a Tweet.
scored_tweets_model_weight_video_playback50: The probability (for a video Tweet) that the user will watch at least half of the video.
scored_tweets_model_weight_reply_engaged_by_author: The probability the user replies to the Tweet and this reply is engaged by the Tweet author.
scored_tweets_model_weight_good_click: The probability the user will click into the conversation of this Tweet and reply or Like a Tweet.
scored_tweets_model_weight_good_click_v2: The probability the user will click into the conversation of this Tweet and stay there for at least 2 minutes.
scored_tweets_model_weight_negative_feedback_v2: The probability the user will react negatively (requesting "show less often" on the Tweet or author, block or mute the Tweet author).
scored_tweets_model_weight_report: The probability the user will click Report Tweet.
```

The outputs of the model are combined into a final model score by doing a weighted sum across the predicted engagement probabilities. 
The weight of each engagement probability comes from a configuration file, read by the serving stack 
[here](https://github.com/twitter/the-algorithm/blob/main/home-mixer/server/src/main/scala/com/twitter/home_mixer/product/scored_tweets/param/ScoredTweetsParam.scala#L84). The exact weights in the file can be adjusted at any time, but the current weighting of probabilities 
(April 5, 2023) is as follows:
```
scored_tweets_model_weight_fav: 0.5
scored_tweets_model_weight_retweet: 1.0
scored_tweets_model_weight_reply: 13.5
scored_tweets_model_weight_good_profile_click: 12.0
scored_tweets_model_weight_video_playback50: 0.005
scored_tweets_model_weight_reply_engaged_by_author: 75.0
scored_tweets_model_weight_good_click: 11.0
scored_tweets_model_weight_good_click_v2: 10.0
scored_tweets_model_weight_negative_feedback_v2: -74.0
scored_tweets_model_weight_report: -369.0
```

Essentially, the formula is:
```
score = sum_i { (weight of engagement i) * (probability of engagement i) }
```

Since each engagement has a different average probability, the weights were originally set so that, 
on average, each weighted engagement probability contributes a near-equal amount to the score. 
Since then, we have periodically adjusted the weights to optimize for platform metrics.

Some disclaimers:
- Due to the need to make sure this runs independently from other parts of Twitter codebase, there may be small differences from the production model.
- We cannot release the real training data due to privacy restrictions. However, we have included a script to generate random data to ensure you can run the model training code.

## Development
After following the repo setup instructions, you can run the following script from a virtual environment to create a 
random training dataset in `$HOME/tmp/recap_local_random_data`:
```sh
projects/home/recap/scripts/create_random_data.sh
```

You can then train the model using the following script.
Checkpoints and logs will be written to `$HOME/tmp/runs/recap_local_debug`:
```sh
projects/home/recap/scripts/run_local.sh
```

The model training can be configured in `projects/home/recap/config/local_prod.yaml`
