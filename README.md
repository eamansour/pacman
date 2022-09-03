# Pac-Man: Markov Decision Process Agent

This is an implementation of an [AI agent](./mdpAgents.py) that is able to play and win Pac-Man through the use of value iteration to solve the Markov Decision Process (MDP).

More interestingly, the AI agent operates in a stochastic environment, as its moves are influenced by probabilities. For each move, the agent has an 80% chance to move in the value-iterated policy direction, and a 20% chance to move in a direction perpendicular to the value-iterated direction.

Original Pac-Man project developed at [UC Berkley](http://ai.berkeley.edu).

## Usage

This project must be run using [Python 2.7](https://www.python.org/download/releases/2.7/).

To run the agent:

```
python pacman.py -p MDPAgent -l <layout>
```

The layouts for the agent's environment can be found in the `layouts` directory. However, the agent was primarily developed to run on the `smallgrid` and `mediumclassic` layouts.

### Other options

Aside from the `-l` option to specify the environment's layout, there are a couple of additional options that can be specified:

- `-q` runs the agent without the UI.
- `-n <number_of_games>` can be used to specify how many Pac-Man games will be executed, where `<number_of_games>` is an integer value.

### Example

The following runs the agent on the `smallgrid` layout for 25 games without the UI:

```
python pacman.py -p MDPAgent -l smallgrid -q -n 25
```
