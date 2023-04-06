# Twitter Machine Learning models

This project open sources some of the ML models used at Twitter.

## Models
Currently these are:

1. The "For You" Heavy Ranker (projects/home/recap).

2. TwHIN embeddings (projects/twhin) https://arxiv.org/abs/2202.05387


## Installation

To install this project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Run the following command to set up a Python virtual environment:

    `./images/init_venv.sh` (Linux only).

Note: This command is only supported on Linux machines.

4. Follow the instructions provided in the READMEs of each project to run the desired model.

### Note:
This project can be run inside a python virtualenv. We have only tried this on Linux machines and because we use `torchrec` it works best with an Nvidia GPU.



## Contributing

To contribute to this project, please follow these steps:

1. Submit a bug report or feature request via the issue tracker.
2. Fork the repository and make your changes on a new branch.
3. Submit a pull request with your changes.

## License

This project is licensed under the [insert license here]. Copyright (c) [insert copyright information].

## Resources

- https://developer.twitter.com/en/docs