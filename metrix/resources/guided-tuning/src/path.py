import os

def get_data_path():
	return os.path.normpath(os.path.dirname(__file__) + '/../data/')

def get_project_root():
	return os.path.normpath(os.path.dirname(__file__) + '/../')