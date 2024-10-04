# Quantitative Analysis and Strategy Creation

Welcome to the Quantitative Analysis and Strategy Creation project. This repository focuses on developing and analyzing various financial strategies using quantitative methods. The primary work is done within the `notebooks` directory, where you will find Jupyter notebooks detailing different strategies and their performance analyses.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project utilizes the Freqtrade framework for creating and automating trading strategies. Freqtrade provides a robust infrastructure for developing, backtesting, and deploying algorithmic trading bots. By leveraging Freqtrade's capabilities, we can efficiently implement, evaluate, and optimize various trading strategies to identify the most promising ones for live trading.

## Project Structure

.
├── notebooks
│   ├── back_testing
│   │   ├── strategy_analysis_example.ipynb
│   │   ├── CVDDivergence_analysis.ipynb
│   ├── black_scholes
│   │   └── hedging_strategies.ipynb
│   ├── kucoin_futures_pairs_analysis.ipynb
│   ├── order_heat_map.ipynb
│   ├── portfolio_analysis.ipynb
│   └── trade_analysis.ipynb
├── data
│   ├── historical_data.csv
│   └── additional_data.csv
├── src
│   ├── strategy.py
│   └── analysis.py
├── strategies
│   ├── __init__.py
│   └── risk
│       └── __init__.py
├── README.md
├── requirements.txt
└── quantative-analysis.code-workspace

- **notebooks/**: Contains Jupyter notebooks for backtesting, strategy analysis, portfolio analysis, and other quantitative analyses.
  - **back_testing/**: Notebooks related to backtesting various strategies.
  - **black_scholes/**: Notebooks focusing on Black-Scholes model and related hedging strategies.
  - Other notebooks for specific analyses like KuCoin futures pairs, order heat maps, and trade analysis.
- **data/**: Directory for storing historical and additional data used in the notebooks.
- **src/**: Source code for strategy implementation and analysis tools.
- **strategies/**: Contains strategy modules and risk management implementations.
- **requirements.txt**: List of dependencies required to run the project.
- **quantative-analysis.code-workspace**: VS Code workspace configuration file.

## Getting Started

### Development Setup
To get started with this project, clone the repository and install the necessary dependencies.

bash
git clone https://github.com/yourusername/quantitative-analysis.git
cd quantitative-analysis
pip install -r requirements.txt

### Running the bot

To run the bot using Docker, follow these steps:

1. **Build the Docker image**:
   
   Navigate to the root directory of the project and build the Docker image using the following command:

   ```bash
   docker build -t quantative-analysis-bot .
   ```

2. **Run the Docker container**:

   Once the image is built, you can run the container with the following command:

   ```bash
   docker compose up
   ```

   This command will:
   - Run the container.
   - Mount the directory from your host to the container.
   - Run the freqtrade command inside `start.sh`

3. **Stopping the container**:

   To stop the running container, use the following command:

   ```bash
   docker compose down
   ```

These steps will help you build and run the bot using Docker, ensuring a consistent and isolated environment for your quantitative analysis and strategy creation.

## Dependencies

The project relies on several Python libraries for data analysis and visualization. The main dependencies are listed in the `requirements.txt` file.

## Usage

Navigate to the `notebooks` directory and open any of the Jupyter notebooks to explore different strategies and their analyses.

bash
cd notebooks
jupyter notebook

## Contributing

We welcome contributions to enhance the project. Please fork the repository and create a pull request with your changes.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.