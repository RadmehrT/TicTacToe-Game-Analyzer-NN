import pandas as pd

class TicTacToeDataProcessor:
    def __init__(self, file_path=None):
        """
        Initialize the class with the path to the CSV file.
        :param file_path: Path to the CSV file.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Load data from the specified file."""
        if self.file_path:
            self.df = pd.read_csv(self.file_path)
        else:
            raise ValueError("File path must be provided.")

    def get_training_inputs(self):
        """Return the input data (without the 'positive' column) as a NumPy array."""
        if self.df is not None:
            
            X = self.df.drop(columns=['positive'])
            return X.to_numpy()
        else:
            raise ValueError("Data is not loaded. Call 'load_data' first.")

    def get_training_outputs(self):
        """Return the 'positive' column as the output labels (as a NumPy array)."""
        if self.df is not None:
            
            y = self.df['positive'].to_numpy()
            return y
        else:
            raise ValueError("Data is not loaded. Call 'load_data' first.")

