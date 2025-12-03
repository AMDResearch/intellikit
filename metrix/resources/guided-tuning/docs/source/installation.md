# Installation Guide

Follow the steps below to install and set up the project:

## Prerequisites

Required Python packages are called out in our `requirements.txt`. In addition, ensure you have the following installed on your system:
- Python `>=3.11`
- ROCm `>=6.5`

## Install

1. Clone the Repository

    ```bash
    git clone https://github.com/AARInternal/guided-tuning.git
    cd guided-tuning
    ```

2. Create a Virtual Environment (Optional)

    It is recommended to use a virtual environment to manage dependencies.

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install Dependencies

    Install the required Python packages:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

4. Test installation (Optional)

    Guided Tuning provides a set of unit tests you can run locally to verify a successful installation. To run the tests, execute the following command:

    ```bash
    python3 -m pip install -r requirements-test.txt
    pytest tests/test_project.py -v
    ```