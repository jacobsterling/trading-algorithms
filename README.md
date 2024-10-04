# Quantitative Analysis

This repository contains notebooks and scripts for quantitative analysis and strategy creation using Freqtrade, a powerful cryptocurrency trading bot.

## Table of Contents

1. [Setup](#setup)
2. [Project Structure](#project-structure)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Customization](#customization)
6. [Integration with start.sh](#integration-with-startsh)
7. [Advanced Usage](#advanced-usage)

## Setup

To set up the project environment, you have two options:

### 1. Using pip and requirements.txt

```
pip install -r requirements.txt
```

This will install all the necessary Python packages listed in the requirements.txt file.

### 2. Using Docker

If you prefer to use Docker, we provide a Dockerfile and docker-compose.yml for easy setup:

```
docker-compose up -d
```

This will build the Docker image and start the container with all required dependencies.

Choose the method that best suits your workflow and system configuration.

## Project Structure

The project is organized as follows:

- `./configuration/`: Contains Python files for different aspects of Freqtrade configuration
- `main.py`: Generates the final configuration
- `start.sh`: Script to automate Freqtrade operations

## Configuration

The `./configuration` directory contains several Python files that define different aspects of the Freqtrade configuration:

- `exchanges.py`: Defines exchange-specific configurations
- `execution.py`: Sets up execution parameters like entry/exit pricing and restrictions
- `pairlists.py`: Defines the pair list configuration
- `server.py`: Configures Telegram and API server settings

The `main.py` file is responsible for generating the final configuration by combining all these components using the `generate_config` function.

## Usage

To generate a configuration for a specific exchange:

1. Ensure your `.env` file is set up with the necessary environment variables (API keys, secrets, etc.).
2. Run the `main.py` script from the command line, specifying the exchange:

This will generate a `config.json` file in the directory specified by `CONFIG_DIR`.

## Customization

To customize the configuration:

1. Modify the relevant files in the `./configuration` directory:
   - Adjust trading pairs in `execution.py`
   - Change exchange settings in `exchanges.py`
   - Modify pair list methods in `pairlists.py`
   - Update Telegram or API server settings in `server.py`
2. If you need to add new configuration options, you can extend the `generate_config` function in `main.py`.

## Integration with start.sh

The `start.sh` script uses this configuration generation process:

This project includes a `start.sh` script to simplify the process of running Freqtrade, a powerful cryptocurrency trading bot. Here's how to use it:

1. Ensure you have Freqtrade installed. If not, follow the installation instructions in the [Freqtrade documentation](https://www.freqtrade.io/en/stable/installation/).

2. Make the script executable (if it's not already):
   ```
   chmod +x start.sh
   ```

3. Run the script:
   ```
   ./start.sh
   ```

The `start.sh` script automates several Freqtrade operations:

- It checks for and creates necessary directories.
- It downloads and updates the latest price data for specified pairs.
- It runs Freqtrade in dry-run mode with the default configuration.

You can customize the script behavior by editing `start.sh`. Some common modifications include:

- Changing the trading pairs
- Adjusting the timeframe for data download
- Modifying Freqtrade parameters

Remember to review and adjust the configuration files in the `user_data` directory to match your trading strategy and preferences.

For more advanced usage and configuration options, refer to the [Freqtrade documentation](https://www.freqtrade.io/en/stable/).



