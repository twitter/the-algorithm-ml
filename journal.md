# Journal <!-- omit in toc -->

- [Initial Prompts](#initial-prompts)
- [Modified Prompt by Claude Opus 4 w/ Extended Thinking](#modified-prompt-by-claude-opus-4-w-extended-thinking)
- [What was sent to Claude code](#what-was-sent-to-claude-code)

## Initial Prompts

<details>

```text

## Context

A couple of years ago, there was a major outcry about the “For You” timeline algorithm not being open source—so Elon’s team open-sourced it.

Since then, no one’s really touched it.

This is a classic example of the difference between open source and public source—and between complaining and actually doing something.

Claude, this is your opportunity to shine and help improve X (formerly known as Twitter).

## What to Do

Take a deep dive into the repository and identify opportunities to improve the “For You” timeline.

Focus on:

- Fairness in free speech
- Minimizing hate speech
- Reducing echo chambers

Building around a user’s actual interests

Expanding those interests in thoughtful ways

## Output

If you find any trivial improvements, go ahead and implement them.

Otherwise, focus on low-hanging high-impact changes to achieve the results above.

Produce a clear, actionable, concise README on next steps.
```

</details>

## Modified Prompt by Claude Opus 4 w/ Extended Thinking

<details>

```text

Analyze Twitter's open-sourced ML algorithm repository to identify concrete improvements for the "For You" timeline that balance free expression with user safety and discovery.

## Context

Twitter open-sourced their recommendation algorithm in 2023. Despite public availability, the community hasn't contributed meaningful improvements. This is your chance to make a real impact on how billions of tweets are ranked and served.

## Primary Analysis Targets

1. **Core Ranking Pipeline** (`src/python/twitter/deepbird/projects/timelines/scripts/models/earlybird/`)

   - Examine the ranking features and scoring mechanisms
   - Look for bias in feature weights that might create echo chambers
   - Identify opportunities to introduce diversity signals

2. **User Representation** (`src/scala/com/twitter/simclusters_v2/`)

   - Analyze how user interests are clustered
   - Find ways to introduce controlled randomness for discovery
   - Look for overfitting to past behavior patterns

3. **Content Safety** (`trust_and_safety_models/`)

   - Review hate speech detection thresholds
   - Identify gaps in harmful content detection
   - Ensure fairness across different viewpoints

4. **Engagement Prediction** (`src/python/twitter/deepbird/projects/timelines/scripts/models/`)
   - Check if engagement metrics overly favor controversial content
   - Look for opportunities to balance engagement with quality signals

## Specific Improvements to Find

**Trivial Changes** (implement immediately):

- Hard-coded constants that could be configurable
- Missing null checks or error handling
- Obvious performance optimizations (e.g., unnecessary loops, repeated calculations)
- Documentation gaps for critical functions

**High-Impact Opportunities**:

- Algorithms that reinforce user bubbles (e.g., similarity thresholds too high)
- Missing diversity injection points in the recommendation pipeline
- Overly aggressive filtering that might silence legitimate viewpoints
- Engagement metrics that don't account for negative externalities

## Deliverables

1. **Code Changes**: Submit 2-3 trivial PRs with immediate improvements
2. **Technical Report** (README.md):
   - Executive summary (3-5 bullet points)
   - Top 5 high-impact changes with:
     - Current behavior analysis
     - Proposed solution
     - Expected impact metrics
     - Implementation complexity (lines of code, dependencies)
   - Architectural diagram showing where changes fit
   - Rough implementation timeline

## Success Metrics

Your recommendations should:

- Increase viewpoint diversity by 15-20%
- Reduce hate speech exposure by 10%
- Maintain or improve user engagement
- Be implementable within 2-4 weeks by a small team

Start by mapping the codebase structure, then dive deep into the ranking pipeline as that's where the most impactful changes likely exist.

```

</details>

## What was sent to Claude code

<details>

```text

Analyze Twitter's open-sourced ML algorithm repository to identify concrete improvements for the "For You" timeline that balance free expression with user safety and discovery.

## Context

Twitter open-sourced their recommendation algorithm in 2023. Despite public availability, the community hasn't contributed meaningful improvements. This is your chance to make a real impact on how billions of tweets are ranked and served.

## Primary Analysis Targets

1. **Core Ranking Pipeline**

   - Examine the ranking features and scoring mechanisms
   - Look for bias in feature weights that might create echo chambers
   - Identify opportunities to introduce diversity signals

2. **User Representation**

   - Analyze how user interests are clustered
   - Find ways to introduce controlled randomness for discovery
   - Look for overfitting to past behavior patterns

3. **Content Safety**

   - Review hate speech detection thresholds
   - Identify gaps in harmful content detection
   - Ensure fairness across different viewpoints

4. **Engagement Prediction**
   - Check if engagement metrics overly favor controversial content
   - Look for opportunities to balance engagement with quality signals

## Specific Improvements to Find

**Trivial Changes** (implement immediately):

- Obvious performance optimizations (e.g., unnecessary loops, repeated calculations)
- Hard-coded constants that could be configurable
- Missing null checks or error handling
- Documentation gaps for critical functions

**High-Impact Opportunities** (low-hanging fruit)

- Algorithms that reinforce user bubbles (e.g., similarity thresholds too high)
- Missing diversity injection points in the recommendation pipeline
- Overly aggressive filtering that might silence legitimate viewpoints
- Engagement metrics that don't account for negative externalities

## Deliverables

1. **Code Changes**: 1 PR w/ immediate trivial improvements
2. **Technical Report** (README.md):
   - Executive summary (3-5 bullet points)
   - Top 5 high-impact changes with:
     - Current behavior analysis
     - Proposed solution
     - Expected impact metrics
     - Implementation complexity (lines of code, dependencies)
   - Architectural mermaid diagram showing where changes fit (use colors and keep it simple)
   - Rough order of operations

## Success Metrics

Your recommendations should aim to:

- Maintain or improve user engagement
- Increase viewpoint diversity by 15-25%
- Reduce hate speech exposure by 10-50%
- Be implementable within 2-4 weeks by a small team

Start by mapping the codebase structure, then dive deep into the ranking pipeline as that's where the most impactful changes likely exist.

```

</details>
